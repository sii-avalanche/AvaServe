import logging
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Optional

import torch

from sglang.srt.distributed import get_pp_group
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.logprob_processor import compute_spec_logprobs
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import _async_d2h
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    compute_position,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSV4_DRAFT_ATTENTION_BACKEND,
    draft_is_deepseek_v4,
    dspark_draft_own_embed_tokens_scope,
    get_dspark_sample_from_anchor,
    read_draft_hf_config,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    make_next_draft_input,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    maybe_build_draft_sampler,
)
from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    TargetHiddenKvInjector,
)
from sglang.srt.speculative.dspark_components.dspark_observability import (
    DsparkStepObservers,
    InfoSegment,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkVerifyPlanner,
    alloc_verify_window,
    dp_global_verify_tier_num_tokens,
    idle_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import (
    CommitInjectCtx,
    DsparkVerifyEpilogue,
    TargetVerifyExecutor,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
    draft_pp_context,
    draft_tp_context,
    prepare_mamba_track_for_verify,
)
from sglang.srt.utils import (
    is_cuda,
    is_cuda_alike,
    is_npu,
    is_pin_memory_available,
)

logger = logging.getLogger(__name__)


_is_npu = is_npu()


@dataclass
class DSparkSamplingCacheEntry:
    """One microbatch's stashed proposal distributions for the sampling accept.

    Folded CUDA-graph proposals alias a replay output buffer, so the caller
    must hand over an owned (cloned) tensor."""

    corrected_logits: torch.Tensor
    rids: list[str]
    seq_lens: list[int]

    def match(
        self, rids: list[str], seq_lens: Optional[list[int]]
    ) -> tuple[Optional[torch.Tensor], Optional[list[Optional[int]]]]:
        """Match rows by (rid, seq_len): the seq_len check rejects stale
        entries (e.g. a retracted + rejoined request with the same rid).
        Returns (logits, None) on an exact full match, (logits, rows) with a
        per-row source index (None = miss) otherwise; (None, None) if no row
        matches at all."""
        cached_rows = {rid: row for row, rid in enumerate(self.rids)}
        rows: list[Optional[int]] = []
        for i, rid in enumerate(rids):
            row = cached_rows.get(rid)
            if (
                row is not None
                and seq_lens is not None
                and self.seq_lens[row] != seq_lens[i]
            ):
                row = None  # stale entry: treat as a miss
            rows.append(row)
        if not any(r is not None for r in rows):
            return None, None
        # The None-rows contract promises the stashed tensor is directly
        # usable by the caller, so it must have exactly the query's row count:
        # after trailing requests finish, the survivors sit at identity
        # positions but the cache carries extra tail rows.
        if (
            len(self.rids) == len(rids)
            and all(r is not None for r in rows)
            and rows == list(range(len(rows)))
        ):
            return self.corrected_logits, None
        return self.corrected_logits, rows


class DSparkSamplingCache:
    """Process-local corrected logits keyed by the PP microbatch slot."""

    def __init__(self):
        self._slots: dict[int, DSparkSamplingCacheEntry] = {}

    def publish(self, mb_id: int, entry: DSparkSamplingCacheEntry) -> None:
        self._slots[mb_id] = entry

    def consume(self, mb_id: int) -> Optional[DSparkSamplingCacheEntry]:
        return self._slots.pop(mb_id, None)

    def clear(self) -> None:
        self._slots.clear()


class DSparkWorkerV2(BaseSpecWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker: TpModelWorker,
        draft_worker_cls: type[TpModelWorker] = TpModelWorker,
    ):
        super().__init__()

        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = get_schedule().page_size
        self.device = target_worker.device

        self._draft_is_moe = draft_is_deepseek_v4()
        self._draft_dp_context_enabled = (
            get_parallel().enable_dp_attention and not self._draft_is_moe
        )
        self._is_pd_prefill = get_disagg().disaggregation_mode == "prefill"
        self._decode_graph_allowed = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
            and not self._is_pd_prefill
        )
        # Only the last PP stage runs the draft model; the other ranks only
        # run the target verify forward and consume the relayed proposal.
        self._hosts_draft = get_pp_group().is_last_rank
        if (
            get_parallel().enable_dp_attention
            and self._draft_is_moe
            and ps.attn_tp_size > 1
        ):
            raise ValueError(
                "DSpark + dp attention with a DeepSeek-V4 (MoE) draft requires "
                "attn_tp == 1 (set --dp-size == --tp). attn_tp > 1 corrupts the "
                "MoE-under-DP all-reduce."
            )

        if self._hosts_draft:
            # The target's embedding lives on the first pipeline stage, so a
            # last-stage draft cannot share it and must load its own copy.
            # Read the pp group before draft_pp_context() patches it away.
            needs_own_embed_tokens = get_pp_group().world_size > 1
            with (
                self._draft_context(),
                draft_pp_context(),
                dspark_draft_own_embed_tokens_scope()
                if needs_own_embed_tokens
                else nullcontext(),
            ):
                bundle = build_draft_tp_worker(
                    server_args=server_args,
                    gpu_id=gpu_id,
                    ps=replace(ps, pp_rank=0, pp_size=1),
                    nccl_port=nccl_port,
                    target_model_config=target_worker.model_runner.model_config,
                    algo_label="DSPARK",
                    attention_backend_override=(
                        DSV4_DRAFT_ATTENTION_BACKEND if self._draft_is_moe else None
                    ),
                    pp_active=needs_own_embed_tokens,
                    draft_worker_cls=draft_worker_cls,
                    # A last-stage-only draft worker cannot enter the WORLD
                    # broadcast; reuse the target's already-broadcast seed.
                    random_seed=target_worker.random_seed,
                )
            self._draft_worker = bundle.draft_worker
            self.draft_model_runner = bundle.draft_model_runner
            self.draft_model = bundle.draft_model
            draft_hf_config = self.draft_model_runner.model_config.hf_config
            if (
                needs_own_embed_tokens
                and self.ps.tp_rank == 0
                and getattr(self.draft_model, "embed_tokens", None) is not None
            ):
                logger.info(
                    "DSpark draft loads its own embed_tokens: the target's "
                    "embedding is on the first pipeline stage."
                )
        else:
            self._draft_worker = None
            self.draft_model_runner = None
            self.draft_model = None
            # Resolve the runtime knobs from the draft checkpoint's own
            # config (the target checkpoint itself when the draft is bundled)
            # without loading draft weights on this rank.
            draft_hf_config = read_draft_hf_config()
        self._draft_sampler = None
        # Per-microbatch stash of the proposal's corrected logits for the
        # sampling accept. Under PP several micro-batches interleave on this
        # worker, so the stash is keyed by the PP microbatch slot (0 without
        # PP): each verify consumes the entry from its own logical batch's
        # previous step, not the latest one overall.
        self._sampling_cache = DSparkSamplingCache()

        # The mask token is input-only (it is embedded, never sampled), so its
        # bound is the embedding-table row count: the PADDED vocab when the
        # target pads its embedding (e.g. Inkling true vocab 200058, padded
        # 201024, mask 200064), else the plain vocab size.
        target_model_config = self.target_worker.model_runner.model_config
        target_embed_rows = (
            getattr(target_model_config.hf_text_config, "padded_vocab_size", None)
            or target_model_config.vocab_size
        )
        self._target_vocab_size = int(target_embed_rows)
        if self._hosts_draft:
            # muP targets declare logits_mup_width_multiplier; the draft was
            # trained against the folded head, so compute_base_logits divides.
            self.draft_model.logits_mup_width_multiplier = getattr(
                target_model_config.hf_text_config, "logits_mup_width_multiplier",
                None,
            )
        self._target_is_mambaish = mambaish_config(target_model_config) is not None
        runtime_config = resolve_runtime_config(
            draft_hf_config=draft_hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=int(target_embed_rows),
        )
        self.gamma = runtime_config.gamma
        self.verify_num_draft_tokens = runtime_config.verify_num_draft_tokens
        self.sample_from_anchor = (
            bool(self.draft_model.sample_from_anchor)
            if self._hosts_draft
            else get_dspark_sample_from_anchor(draft_hf_config)
        )
        self.query_token_num = self.gamma if self.sample_from_anchor else self.gamma + 1
        self.speculative_num_draft_tokens = self.verify_num_draft_tokens
        self._mask_token_id = runtime_config.mask_token_id

        parallel = get_parallel()
        self._tp_sync = SpecTpSync(
            parallel.attn_tp_group
            if parallel.enable_dp_attention
            else parallel.tp_group
        )
        self._draft_graph_group = (
            parallel.attn_tp_group
            if self._draft_dp_context_enabled
            else parallel.tp_group
        )

        if self.ps.tp_rank == 0 and self._hosts_draft:
            logger.info(
                "Initialized DSpark draft runner. attention_backend=%s, model=%s, "
                "gamma=%s, verify_num_draft_tokens=%s, query_token_num=%s, "
                "sample_from_anchor=%s, mask_token_id=%s, markov_head=%s",
                bundle.resolved_attention_backend,
                self.draft_model.__class__.__name__,
                self.gamma,
                self.verify_num_draft_tokens,
                self.query_token_num,
                self.sample_from_anchor,
                self._mask_token_id,
                type(self.draft_model.markov_head).__name__,
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.verify_num_draft_tokens, device=self.device
        )
        if self._hosts_draft:
            self._draft_block_spec_info = make_draft_block_spec_info(
                draft_token_num=int(self.query_token_num), device=self.device
            )

            if getattr(self.draft_model, "uses_own_vocab_modules", False):
                if self.ps.tp_rank == 0:
                    logger.info(
                        "DSpark draft uses its checkpoint-local embedding and LM head."
                    )
            else:
                target_model = self.target_worker.model_runner.model
                lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
                if lm_head is None or not hasattr(lm_head, "weight"):
                    raise RuntimeError(
                        "DSpark requires the target model to expose `lm_head` with `weight`."
                    )
                self.draft_model.attach_shared_modules(
                    embed_tokens=unwrap_lora_layer(
                        self._resolve_target_embed_tokens(target_model)
                    ),
                    lm_head=lm_head,
                )

            self._verify_planner = DSparkVerifyPlanner(
                draft_model=self.draft_model,
                gamma=self.gamma,
                model_runner=self.model_runner,
                device=self.device,
                tp_rank=self.ps.tp_rank,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                tp_sync=self._tp_sync,
            )
            if (
                get_parallel().enable_dp_attention
                and not self._draft_is_moe
                and self._verify_planner.is_compact_mode
                and self._decode_graph_allowed
            ):
                raise ValueError(
                    "DSpark dense-draft compact verify under --enable-dp-attention does not "
                    "yet support cuda graph (idle DP groups cannot join the token-keyed "
                    "compact graph). Re-run with --disable-cuda-graph (eager is lossless), "
                    "or use SGLANG_RAGGED_VERIFY_MODE=static. The dsv4 (MoE) draft supports "
                    "cuda graph under DP."
                )
            self._kv_injector = TargetHiddenKvInjector(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                model_runner=self.model_runner,
                device=self.device,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                block_pos_offsets=self._block_pos_offsets,
            )
            self._proposer = DraftBlockProposer(
                draft_model=self.draft_model,
                draft_model_runner=self.draft_model_runner,
                gamma=self.gamma,
                mask_token_id=self._mask_token_id,
                draft_block_spec_info=self._draft_block_spec_info,
                tp_sync=self._tp_sync,
                dp_moe_sync=self._draft_is_moe and get_parallel().enable_dp_attention,
            )
        else:
            self._draft_block_spec_info = None
            self._verify_planner = None
            self._kv_injector = None
            self._proposer = None
        self._verify_epilogue = None
        if (
            self._hosts_draft
            and self._verify_planner.is_compact_mode
            and self._decode_graph_allowed
            and is_cuda()
        ):
            self._verify_epilogue = DsparkVerifyEpilogue(
                max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                device=self.device,
                tp_sync=self._tp_sync,
                commit_ctx=CommitInjectCtx(
                    draft_model=self.draft_model,
                    block_pos_offsets=self._block_pos_offsets,
                    resolve_pool=lambda: self.draft_model_runner.token_to_kv_pool,
                    resolve_req_to_token=lambda: (
                        self.model_runner.req_to_token_pool.req_to_token
                    ),
                ),
            )
            self.model_runner.capture_tail_hooks.append(
                self._verify_epilogue.capture_hook
            )

        self._simulate_acc_len = float(envs.SGLANG_SIMULATE_ACC_LEN.get())
        if (
            self._simulate_acc_len > 0
            and self._simulate_acc_len != 1.0
            and self._hosts_draft
            and not self._verify_planner.is_verify_all
        ):
            raise ValueError(
                "SGLANG_SIMULATE_ACC_LEN>1.0 with DSpark requires a verify-all "
                "schedule (SGLANG_RAGGED_VERIFY_MODE=static, or =compact with the "
                "uninitialized/flat SPS table): a constant simulated correct_len>0 "
                "can exceed a trimmed request's verify budget (cap-accept, or "
                "compact with a profiled SPS table) and break the cutoff/cap "
                "accounting. SGLANG_SIMULATE_ACC_LEN=1.0 yields correct_len=0 "
                "(commit is the bonus token only), which stays within every verify "
                "budget and is safe in any mode. Got mode="
                f"{self._verify_planner.mode_value!r}, simulate_acc_len="
                f"{self._simulate_acc_len}."
            )

        self._verify_executor = TargetVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,
            tp_sync=self._tp_sync,
            verify_epilogue=self._verify_epilogue,
            simulate_acc_len=self._simulate_acc_len,
        )

        self._forced_budget_frac: Optional[float] = None
        self._need_mamba_verify_commit = False

        if self._hosts_draft:
            self._observers = DsparkStepObservers(
                planner=self._verify_planner,
                gamma=self.gamma,
                verify_num_draft_tokens=self.verify_num_draft_tokens,
                tp_rank=self.ps.tp_rank,
                device=self.device,
                simulate_acc_len=self._simulate_acc_len,
            )
        else:
            self._observers = None

        if self._is_pd_prefill and not self._draft_is_moe:
            self.draft_model.prune_to_ctx_kv_injection()

    def _resolve_target_embed_tokens(self, target_model):
        if hasattr(target_model, "get_input_embeddings"):
            return target_model.get_input_embeddings()
        return target_model.model.get_input_embeddings()

    @property
    def carries_confidence(self) -> bool:
        return self._hosts_draft and self._verify_planner.carries_confidence

    @property
    def spec_v2_attn_backends(self) -> tuple:
        if not self._hosts_draft:
            return (self._target_worker.model_runner.attn_backend,)
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def __getattr__(self, name):
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def _draft_context(self):
        if self._draft_dp_context_enabled:
            return draft_tp_context(get_parallel().attn_tp_group)
        return nullcontext()

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if not self._hosts_draft:
            return
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if not self._hosts_draft:
            return
        with self._draft_context():
            self._draft_worker.init_attention_backends()
        self._need_mamba_verify_commit = mambaish_config(
            self.model_runner.model_config
        ) is not None and hasattr(
            self.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )

    def init_cuda_graphs(self):
        if not self._hosts_draft:
            return
        capture_decode_cuda_graph = self._decode_graph_allowed
        available_mem = self._tp_sync.available_memory_gb(
            SpecTpSyncSite.DSPARK_MEM,
            self.device,
            self.gpu_id,
            group=self._draft_graph_group,
        )
        if is_cuda_alike() and capture_decode_cuda_graph:
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DSpark draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        with self._draft_context():
            if capture_decode_cuda_graph:
                # Keep the draft model graph enabled when folded proposal is
                # disabled, but do not capture the proposal head as a tail
                # hook. The proposer will compute base logits and the Markov
                # block eagerly from the graph's hidden states instead. Apart
                # from being the intended precision fallback, skipping the
                # unused hook avoids paying for two proposal computations.
                if envs.SGLANG_DSPARK_FOLDED_PROPOSAL.get():
                    self._draft_sampler = self._maybe_build_draft_sampler(
                        available_memory_gb=available_mem
                    )
                    if self._draft_sampler is not None:
                        self.draft_model_runner.capture_tail_hooks.append(
                            make_draft_sampler_capture_hook(self._draft_sampler)
                        )
                self._proposer.attach_draft_sampler(self._draft_sampler)
            self._draft_worker.init_cuda_graphs(
                capture_decode_cuda_graph=capture_decode_cuda_graph
            )

    def _maybe_build_draft_sampler(self, *, available_memory_gb: float):
        return maybe_build_draft_sampler(
            draft_model=self.draft_model,
            gamma=self.gamma,
            max_bs=max(get_exec().graph.cuda_graph_config.decode.bs),
            device=self.device,
            tp_rank=self.ps.tp_rank,
            tp_sync=self._tp_sync,
            available_memory_gb=available_memory_gb,
            confidence_fn=(
                self._verify_planner.compute_confidence_tensor
                if self._verify_planner.carries_confidence
                else None
            ),
            out=(
                self._verify_epilogue.draft_tokens_buf
                if self._verify_epilogue is not None
                else None
            ),
        )

    def clear_cache_pool(self):
        self._sampling_cache.clear()

    def set_dspark_forced_budget_frac(self, frac: Optional[float]) -> None:
        self._forced_budget_frac = frac
        if not self._hosts_draft:
            return
        self._verify_planner.set_forced_budget_frac(frac)

    def dump_info_records(self) -> Optional[dict]:
        if not self._hosts_draft:
            return None
        return self._observers.dump_info_records()

    def clear_info_records(self) -> None:
        if not self._hosts_draft:
            return
        self._observers.clear_info_records()

    def block_accept_estimate_log_suffix(self) -> Optional[str]:
        if not self._hosts_draft:
            return None
        return self._observers.block_accept_estimate_log_suffix()

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        if not self._hosts_draft:
            return
        self._observers.note_request_finished(rid=rid, natural_stop=natural_stop)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if self._hosts_draft:
                self._verify_planner.note_non_decode_step()
                self._observers.note_prefill_step()
            return self._forward_prefill(batch, on_publish, pp_proxy_tensors)

        return self._forward_decode(batch, on_publish, grammar_barrier, pp_proxy_tensors)

    def _forward_prefill(
        self, batch: ScheduleBatch, on_publish, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_idle():
            if get_parallel().enable_dp_attention:
                self.target_worker.forward_batch_generation(
                    batch,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            return self._decode_idle_result(on_publish=on_publish)

        batch_output = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        if next_token_ids is not None:
            # Non-last PP ranks run the prefill forward without sampling.
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = batch.seq_lens
        batch_output.new_seq_lens = new_seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)

        if not self._hosts_draft:
            # The draft model lives on the last PP stage; this rank only runs
            # the target's share of the pipeline forward.
            return batch_output

        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target aux hidden capture for prefill, but got None. "
                "Make sure the target model has DFlash layers-to-capture configured."
            )
        if batch.extend_lens is None or batch.prefix_lens is None:
            raise RuntimeError(
                "DSpark expected extend_lens / prefix_lens in extend mode, got None."
            )
        if batch.out_cache_loc is None:
            raise RuntimeError("DSpark prefill expected out_cache_loc, but got None.")

        # Must inject before prefill returns: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        device = next_token_ids.device
        pin_memory = is_pin_memory_available(device)
        ctx_lens = torch.tensor(
            batch.extend_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        draft_seq_lens = torch.tensor(
            batch.prefix_lens, dtype=torch.int32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        positions, _ = compute_position(
            self.model_runner.prefill_attention_backend_str,
            draft_seq_lens,
            ctx_lens,
            int(sum(batch.extend_lens)),
        )
        # unified_kv injects into the SWA ring keyed by (draft req slot, position);
        # thread the per-token state_slot + the req's final position so the
        # injector keeps only the last SWA window (older prefill tokens share a
        # ring slot and would race). Cheap; only consumed under unified_kv.
        state_slot = final_pos = None
        if is_unified_kv_triton():
            repeats = ctx_lens.to(torch.int64)
            state_slot = torch.repeat_interleave(
                batch.req_pool_indices.to(device=device, dtype=torch.int64), repeats
            )
            final_pos = torch.repeat_interleave(
                (draft_seq_lens + ctx_lens - 1).to(torch.int64), repeats
            )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=positions,
            state_slot=state_slot,
            final_pos=final_pos,
        )
        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        batch_output.next_draft_input = make_next_draft_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=new_seq_lens,
        )
        batch_output.next_draft_input.mask_token_id = self._mask_token_id
        return batch_output

    def _idle_verify_ragged_layout(self, batch: ScheduleBatch):
        if batch.global_num_tokens is None or not self._verify_planner.is_compact_mode:
            return None
        global_bs = max(batch.global_num_tokens)
        if global_bs <= 0:
            return None
        return idle_ragged_layout(
            tier_num_reqs=global_bs,
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
        )

    def _dp_verify_tier_num_tokens(self, batch: ScheduleBatch) -> Optional[int]:
        if not (
            self._draft_is_moe
            and get_parallel().enable_dp_attention
            and batch.global_num_tokens is not None
            and self._verify_planner.is_compact_mode
        ):
            return None
        return dp_global_verify_tier_num_tokens(
            global_tier_num_tokens=batch.global_spec_verify_tier_num_tokens
        )

    def _decode_idle_result(
        self,
        *,
        on_publish,
    ) -> GenerationBatchResult:
        next_draft_input = make_next_draft_input(
            bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
        )
        next_draft_input.mask_token_id = self._mask_token_id
        if on_publish is not None:
            on_publish(next_draft_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=torch.empty((0,), dtype=torch.int64, device=self.device),
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=next_draft_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=next_draft_input.new_seq_lens,
        )

    def _forward_decode(
        self, batch: ScheduleBatch, on_publish, grammar_barrier=None, pp_proxy_tensors=None
    ) -> GenerationBatchResult:
        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            if self._hosts_draft:
                self._observers.note_idle_decode_step()
                if get_parallel().enable_dp_attention:
                    # Mirror the active decode order (target verify first, then
                    # draft rollout). The EP MoE collectives of busy and idle
                    # ranks pair positionally on the shared DeepEP buffer, so
                    # running the draft dummy first permutes the pairing and
                    # deadlocks the busy ranks' next forward.
                    self._verify_executor.run_idle_participation(
                        batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
                    )
                    if self._draft_is_moe:
                        self._proposer.run_idle_participation(batch)
            return self._decode_idle_result(on_publish=on_publish)

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        sampling_info = batch.sampling_info
        pending = draft_input.pending_draft_tokens
        if pending is None:
            if get_parallel().enable_dp_attention:
                # Under DP attention, idle peers mirror every decode step with
                # verify-shaped dummies; a width-1 plain decode would break the
                # per-rank token accounting (idle mirrors scale global counts by
                # the verify width) and deadlock the EP collectives. Route the
                # step through the verify path with an all-mask proposal
                # instead: masks are never accepted, so the step emits exactly
                # the bonus token, and the accept path falls back to
                # target-only sampling for the missing draft logits.
                pending = torch.full(
                    (bs, self.gamma),
                    self._mask_token_id,
                    dtype=torch.int64,
                    device=device,
                )
            else:
                # No proposal available: the first decode step after a
                # (re)prefill or a mixed batch. Run a plain target decode
                # instead of a verify; the draft host proposes for the next
                # step at the end.
                return self._forward_plain_decode(
                    batch, on_publish, draft_input, pp_proxy_tensors
                )
        draft_tokens = pending
        # Garbage token ids (uninitialized/stale proposal buffer slots, e.g.
        # the draft sampler's torch.empty static output buffer rows a folded
        # replay did not refresh) must not reach the target embedding, the
        # draft-logits scatter in _aligned_draft_block, or the accept path:
        # with dp-lm-head the embedding runs its tp_size==1 branch, which does
        # no shard masking, so an out-of-range id triggers a device-side
        # assert in F.embedding and kills every scheduler. Draft tokens are
        # only hints -- wrong ones are rejected by the accept path -- so remap
        # out-of-range ids to the mask token (never accepted), preserving
        # verify semantics. (masked_fill returns a new tensor; the aliased
        # proposal buffer is left untouched.)
        draft_tokens = draft_tokens.masked_fill(
            (draft_tokens < 0) | (draft_tokens >= self._target_vocab_size),
            int(self._mask_token_id),
        )

        verify_window = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )
        verify_ids_2d = torch.cat(
            [draft_input.bonus_tokens.view(bs, 1), draft_tokens], dim=1
        ).contiguous()
        # Garbage token ids (uninitialized/stale proposal buffer slots, e.g.
        # the draft sampler's torch.empty static output buffer rows a folded
        # replay did not refresh) must never reach the target embedding: with
        # dp-lm-head the embedding runs its tp_size==1 branch, which does no
        # shard masking, so an out-of-range id triggers a device-side assert
        # in F.embedding and kills every scheduler. Draft tokens are only
        # hints -- wrong ones are rejected by the accept path -- so remap
        # out-of-range ids to the mask token (never accepted), preserving
        # verify semantics.
        verify_ids_2d = verify_ids_2d.masked_fill(
            (verify_ids_2d < 0) | (verify_ids_2d >= self._target_vocab_size),
            int(self._mask_token_id),
        )

        # Must stay ahead of the target verify launch below.
        grammar_tree = (
            GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
        )

        prepare_mamba_track_for_verify(batch)
        target_verify = self._verify_executor.run_non_compact(
            batch=batch,
            draft_input=draft_input,
            verify_ids_2d=verify_ids_2d,
            verify_window=verify_window,
            sampling_info=sampling_info,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self._hosts_draft:
            # Acceptance, draft-KV commit and the next proposal are produced
            # on the last PP stage; the results reach this rank through the
            # PP output ring.
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=None,
                next_draft_input=None,
                can_run_cuda_graph=target_verify.can_run_cuda_graph,
                pp_hidden_states_proxy_tensors=target_verify.pp_proxy_tensors,
            )

        self._observers.begin_step()
        logits_output = target_verify.logits_output
        can_run_cuda_graph = target_verify.can_run_cuda_graph
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
            if grammar_mask is not None:
                grammar_mask.apply(logits_output.next_token_logits)

        # The full proposal (with the draft-side logits the accept needs for
        # sampling) was stashed at the end of the previous step; align its
        # rows to the current batch composition.
        draft_block = self._aligned_draft_block(batch, draft_input, draft_tokens)
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=False,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=None,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
        )
        staged_seq_lens = self._stage_seq_lens_d2h(accept.new_seq_lens)
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                accept.out_tokens.reshape(-1),
                chain_stride=self.verify_num_draft_tokens,
            )

        if on_publish is not None:
            on_publish(accept.new_seq_lens)

        self._commit_target_mamba_states_after_verify(
            batch=batch,
            seq_lens_pre_verify=prefix_lens,
            seq_lens_post_verify=accept.new_seq_lens,
            commit_lens=accept.commit_lens,
        )

        self._verify_executor.commit_hidden(
            batch=batch,
            layout=None,
            hidden_strided=None,
            verify_window=verify_window,
            logits_output=logits_output,
            commit_lens=accept.commit_lens,
            bs=bs,
            run_compact=False,
        )
        logits_output.hidden_states = None

        self._observers.observe_verify_step(
            forward_ct=int(batch.forward_iter),
            reqs=batch.reqs,
            bs=bs,
            proposal_folded=False,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            layout=None,
            confidence=None,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            draft_block=draft_block,
            sampling_info=sampling_info,
            correct_len=accept.correct_len,
            cap_trim_lens=accept.cap_trim_lens,
            bonus=accept.bonus,
            commit_lens=accept.commit_lens,
            verify_token_budget=None,
            req_pool_indices=batch.req_pool_indices,
            verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
            dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
        )

        next_draft_input = self._propose_next(
            batch=batch,
            bonus=accept.bonus,
            new_seq_lens=accept.new_seq_lens,
            sampling_info=sampling_info,
            staged_seq_lens=staged_seq_lens,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.out_tokens.reshape(-1),
            accept_lens=accept.commit_lens,
            block_accept_lens=accept.commit_lens + accept.cap_trim_lens,
            cap_lens=None,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
            new_seq_lens=accept.new_seq_lens,
        )

    def _forward_plain_decode(
        self,
        batch: ScheduleBatch,
        on_publish,
        draft_input: DFlashDraftInputV2,
        pp_proxy_tensors=None,
    ) -> GenerationBatchResult:
        """A decode step without a usable proposal: run a plain target decode
        (one sampled token per request), then let the draft host propose for
        the next step. Used on the first decode after a (re)prefill or a
        mixed batch, where no proposal was produced beforehand."""
        bs = len(batch.seq_lens)
        device = self.device
        prefix_lens = batch.seq_lens

        batch.input_ids = draft_input.bonus_tokens.view(-1)
        batch.out_cache_loc = alloc_verify_window(
            batch=batch,
            bs=bs,
            device=device,
            verify_num_draft_tokens=1,
            block_pos_offsets=self._block_pos_offsets[:1],
            model_runner=self.model_runner,
        ).verify_cache_loc
        batch_output = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self._hosts_draft:
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=None,
                next_draft_input=None,
                can_run_cuda_graph=batch_output.can_run_cuda_graph,
                pp_hidden_states_proxy_tensors=(
                    batch_output.pp_hidden_states_proxy_tensors
                ),
            )

        next_token_ids = batch_output.next_token_ids
        if next_token_ids is not None:
            self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, next_token_ids)
        new_seq_lens = prefix_lens + 1
        staged_seq_lens = self._stage_seq_lens_d2h(new_seq_lens)
        if on_publish is not None:
            on_publish(new_seq_lens)

        logits_output = batch_output.logits_output
        if logits_output.hidden_states is None:
            raise RuntimeError(
                "DSpark requires target hidden capture for the plain-decode "
                "fallback, but got None."
            )
        self._kv_injector.inject_target_hidden(
            target_hidden=logits_output.hidden_states,
            cache_loc=batch.out_cache_loc,
            positions=prefix_lens,
            state_slot=None,
            final_pos=None,
        )
        logits_output.hidden_states = None

        next_draft_input = self._propose_next(
            batch=batch,
            bonus=next_token_ids,
            new_seq_lens=new_seq_lens,
            sampling_info=batch.sampling_info,
            staged_seq_lens=staged_seq_lens,
        )

        stride = int(self.verify_num_draft_tokens)
        out_tokens = torch.zeros(
            (bs, stride), dtype=next_token_ids.dtype, device=device
        )
        out_tokens[:, 0] = next_token_ids
        accept_lens = torch.ones((bs,), dtype=torch.int32, device=device)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=accept_lens,
            block_accept_lens=accept_lens,
            cap_lens=None,
            can_run_cuda_graph=batch_output.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=stride,
            new_seq_lens=new_seq_lens,
        )

    def _stage_seq_lens_d2h(self, new_seq_lens: torch.Tensor):
        """Enqueue an async D2H of post-step seq lens right after the
        producing kernels, so consuming it in `_propose_next` only waits for
        those kernels instead of everything queued on the stream by then
        (commit_hidden, mamba commits, draft prep)."""
        cpu_mirror = _async_d2h(new_seq_lens)
        done_event = torch.cuda.Event()
        done_event.record()
        return done_event, cpu_mirror

    def _propose_next(
        self,
        *,
        batch: ScheduleBatch,
        bonus: torch.Tensor,
        new_seq_lens: torch.Tensor,
        sampling_info,
        staged_seq_lens,
    ) -> DFlashDraftInputV2:
        """Compute the proposal for the NEXT verify step on the draft host and
        pack it into the next draft input (relayed to the other PP ranks
        through the output ring)."""
        next_draft_input = make_next_draft_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
        )
        next_draft_input.mask_token_id = self._mask_token_id
        if not self._hosts_draft:
            return next_draft_input

        # The next verify's window gathers slots reserved by the scheduler's
        # per-step over-allocation; the draft rollout writes its KV there.
        batch.seq_lens = new_seq_lens
        done_event, cpu_mirror = staged_seq_lens
        done_event.synchronize()
        batch.seq_lens_cpu = cpu_mirror
        next_window = alloc_verify_window(
            batch=batch,
            bs=len(new_seq_lens),
            device=self.device,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            block_pos_offsets=self._block_pos_offsets,
            model_runner=self.model_runner,
        )
        target_model = self.target_worker.model_runner.model
        with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
            proposal = self._proposer.propose(
                batch=batch,
                draft_input=next_draft_input,
                verify_window=next_window,
                bs=len(new_seq_lens),
                device=self.device,
                target_model=target_model,
                sampling_info=sampling_info,
            )
        draft_tokens = proposal.draft_block.draft_tokens
        if proposal.folded:
            # The folded proposal aliases the draft sampler's static CUDA-graph
            # output buffer, which the next draft replay overwrites. Under PP
            # several micro-batches are in flight, so the relayed proposal must
            # own its storage to stay intact until the next verify reads it.
            draft_tokens = draft_tokens.clone()
        next_draft_input.pending_draft_tokens = draft_tokens
        corrected_logits = proposal.draft_block.corrected_logits
        if corrected_logits is not None:
            if proposal.folded:
                # Same static-buffer aliasing as draft_tokens above.
                corrected_logits = corrected_logits.clone()
            self._sampling_cache.publish(
                getattr(batch, "pp_mb_id", None) or 0,
                DSparkSamplingCacheEntry(
                    corrected_logits=corrected_logits,
                    rids=[r.rid for r in batch.reqs],
                    seq_lens=batch.seq_lens_cpu.tolist(),
                ),
            )
        return next_draft_input

    def _aligned_draft_block(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        draft_tokens: torch.Tensor,
    ):
        from sglang.srt.speculative.dspark_components.dspark_draft import (
            DraftBlockResult,
            resolve_greedy_mask,
        )

        """Rebuild the accept-time draft_block for this step: the draft-token
        ids come from the (already batch-aligned) pending proposal; the
        draft-side logits come from this microbatch's sampling cache, matched
        by (request id, seq_len) so stale rows fall back to target-only
        sampling."""

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        greedy_mask = resolve_greedy_mask(
            bs=bs, sampling_info=sampling_info, device=self.device
        )
        any_sampling = sampling_info is not None and not sampling_info.is_all_greedy
        if sampling_info is None:
            temperatures = torch.ones(bs, dtype=torch.float32, device=self.device)
        else:
            temperatures = (
                sampling_info.temperatures.view(-1).to(torch.float32).clamp_min(1e-5)
            )

        if not any_sampling:
            # Greedy accept compares token ids only.
            return DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=None,
                greedy_mask=greedy_mask,
                temperatures=temperatures,
            )
        entry = self._sampling_cache.consume(getattr(batch, "pp_mb_id", None) or 0)
        stashed_logits, rows = (
            entry.match(
                [r.rid for r in batch.reqs],
                seq_lens=(
                    batch.seq_lens_cpu.tolist()
                    if batch.seq_lens_cpu is not None
                    else None
                ),
            )
            if entry is not None
            else (None, None)
        )
        if stashed_logits is None:
            # No stashed logits (first step after a (re)prefill): fall back to
            # exact target-only sampling. Putting the spike on the actual
            # candidates makes q a delta on the candidate, so the rejection
            # rule accepts it with prob p(candidate) and otherwise resamples
            # from (p-q)+ -- the emitted distribution stays exactly the target
            # distribution (the draft only loses its acceleration). Spiking
            # the mask token instead would make q(candidate) ~ 0 and accept
            # every draft unconditionally, corrupting the output distribution.
            corrected_logits = torch.zeros(
                (bs, self.gamma, self._target_vocab_size),
                dtype=torch.float32,
                device=self.device,
            )
            corrected_logits.scatter_(
                2, draft_tokens.clamp(min=0).unsqueeze(-1), 1e9
            )
            return DraftBlockResult(
                draft_tokens=draft_tokens,
                corrected_logits=corrected_logits,
                greedy_mask=greedy_mask,
                temperatures=temperatures,
            )

        if rows is None:
            corrected_logits = stashed_logits
        else:
            hit = torch.tensor(
                [r is not None for r in rows], dtype=torch.bool, device=self.device
            )
            src_rows = torch.tensor(
                [r if r is not None else 0 for r in rows],
                dtype=torch.long,
                device=self.device,
            )
            corrected_logits = torch.zeros(
                (bs, self.gamma, self._target_vocab_size),
                dtype=stashed_logits.dtype,
                device=self.device,
            )
            # Miss rows fall back to exact target-only sampling (q = delta on
            # the candidate), same as the no-stash path above.
            corrected_logits.scatter_(
                2, draft_tokens.clamp(min=0).unsqueeze(-1), 1e9
            )
            if hit.any():
                corrected_logits[hit] = stashed_logits[src_rows[hit]]
        return DraftBlockResult(
            draft_tokens=draft_tokens,
            corrected_logits=corrected_logits,
            greedy_mask=greedy_mask,
            temperatures=temperatures,
        )

    def _commit_target_mamba_states_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        seq_lens_post_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit the last accepted verify step's KDA/mamba state (chain
        layout: step index = commit_lens - 1) into the persistent caches."""
        if not self._need_mamba_verify_commit:
            return
        # Chain layout only: step index = commit_lens - 1. A tree (topk > 1)
        # layout would need the accept-index mapping the shared spec_utils
        # commit helper does.
        assert get_spec().speculative_eagle_topk in (None, 1)
        attn_backend = self.target_worker.model_runner.attn_backend

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = mamba_track_grid(batch.tree_cache.page_size)
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
            req_pool_indices=batch.req_pool_indices,
        )

    def get_confidence_budget_prepare(self):
        return self._verify_planner.confidence_budget_prepare()
