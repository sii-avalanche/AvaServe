from typing import Any

import regex as re
import torch
from humming.layer import HummingInputSchema, HummingMethod
from humming.schema import BaseWeightSchema

from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.runtime_context import get_exec


def humming_is_layer_skipped(config: dict[str, Any], prefix: str):
    if not config:
        return True

    keys = ["ignored_layers", "ignore", "modules_to_not_convert"]
    ignored_layers: list[str] = []
    for key in keys:
        ignored_layers = config.get(key, []) or []
        if not ignored_layers:
            break

    if any(module_name in prefix for module_name in ignored_layers):
        return True
    if "lm_head" in prefix:
        return True

    for regex in config.get("dynamic", {}):
        if regex[:1] != "-":
            continue
        if re.match(regex[2:], prefix):
            return True

    return False


def _set_humming_dispatcher_output_dtype(
    layer: torch.nn.Module, output_dtype: str
) -> None:
    dispatcher = getattr(layer, "dispatcher", None)
    if dispatcher is None:
        return

    quant_config = dict(getattr(dispatcher, "quant_config", None) or {})
    quant_config["dispatcher_output_dtype"] = output_dtype
    dispatcher.set_quant_config(quant_config)


def configure_humming_deepep_dispatch(layer: torch.nn.Module) -> bool:
    if not get_moe_a2a_backend().is_deepep():
        layer._humming_uses_deepep_fp8_dispatch = False
        _set_humming_dispatcher_output_dtype(layer, "bf16")
        return False

    output_dtype = get_exec().moe.deepep_dispatcher_output_dtype
    if output_dtype == "auto":
        output_dtype = "bf16" if envs.SGLANG_DEEPEP_BF16_DISPATCH.get() else "fp8"
    if output_dtype not in ("bf16", "fp8"):
        raise ValueError(
            f"Humming does not support DeepEP {output_dtype} dispatch; "
            "use --deepep-dispatcher-output-dtype=bf16 or fp8."
        )

    _set_humming_dispatcher_output_dtype(layer, output_dtype)
    use_fp8 = output_dtype == "fp8"
    layer._humming_uses_deepep_fp8_dispatch = use_fp8
    return use_fp8


# Expert-dim chunk size for the staged MoE weight transform. Only one
# chunk's repack output lives on the GPU at a time; per-sublayer results
# accumulate in pinned host memory and move back after the originals are
# freed. This caps the load-time GPU peak at ~one chunk instead of a full
# sublayer copy (the default whole-layer transform allocates its entire
# int32 repack output while all raw weights are still resident, which is
# what OOMed TP=1/PP=8 and 17-layer TP=2 stages).
_STAGED_TRANSFORM_CHUNK_EXPERTS = 16


class _ExpertChunkProxy:
    """Presents one expert slice of an MoE sublayer as a standalone layer.

    Lets humming's own transform run unmodified on the chunk; the caller
    only swaps in a chunk-sized ``num_experts`` meta (scale fusion reshapes
    by it) and harvests the output parameters afterwards.
    """

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self._tensors = tensors

    def state_dict(self):
        return self._tensors


def _transform_moe_sublayer_staged(
    layer: torch.nn.Module,
    sublayer_name: str,
    meta_kwargs: dict,
    chunk_experts: int = _STAGED_TRANSFORM_CHUNK_EXPERTS,
) -> bool:
    """Memory-frugal variant of HummingMethod.transform_humming_layer.

    Runs the identical per-expert transform in expert-dim chunks (layout
    repack, padding and scale fusion are all per-expert independent), moving
    each chunk's outputs to pinned host memory. The original full-size
    weights are deleted only after every chunk is processed, and the final
    parameters are assembled back on the GPU afterwards -- so the transient
    GPU overhead is one chunk instead of a whole second copy of the sublayer.
    Returns False when the sublayer is not expert-stacked and the caller
    must fall back to the whole-layer transform.
    """
    prefix = layer.humming_metas[sublayer_name].name_prefix
    orig = {
        name: param
        for name, param in layer.named_parameters()
        if name.startswith(prefix)
    }
    num_experts = meta_kwargs["num_experts"]
    if not num_experts or not any(
        t.ndim >= 1 and t.shape[0] == num_experts for t in orig.values()
    ):
        return False

    device = next(iter(orig.values())).device
    staged: dict[str, list[tuple[int, int, torch.Tensor]]] = {}
    passthrough: dict[str, torch.Tensor] = {}
    for start in range(0, num_experts, chunk_experts):
        stop = min(start + chunk_experts, num_experts)
        chunk_tensors = {
            name: (
                tensor[start:stop]
                if tensor.ndim >= 1 and tensor.shape[0] == num_experts
                else tensor
            )
            for name, tensor in orig.items()
        }
        proxy = _ExpertChunkProxy(chunk_tensors)
        HummingMethod.prepare_layer_meta(
            proxy, **{**meta_kwargs, "num_experts": stop - start}
        )
        HummingMethod.transform_humming_layer(proxy, sublayer_name=sublayer_name)
        outputs = {
            name: value.data
            for name, value in vars(proxy).items()
            if isinstance(value, torch.nn.Parameter)
        }
        if not all(
            out.ndim >= 1 and out.shape[0] == stop - start for out in outputs.values()
        ):
            # Unexpected non-expert-leading output; cannot stitch safely.
            return False
        for name, out in outputs.items():
            pinned = torch.empty(out.shape, dtype=out.dtype, pin_memory=True)
            pinned.copy_(out, non_blocking=False)
            staged.setdefault(name, []).append((start, stop, pinned))
        del proxy, chunk_tensors, outputs

    # Every chunk repacked: drop the originals first so their storage can be
    # reused by the assembly allocations below. Parameters the transform did
    # not produce (e.g. an optional zero_point the config disables) are put
    # back verbatim, mirroring what the whole-layer transform would keep.
    for name in orig:
        delattr(layer, name)
    for name, param in orig.items():
        if name not in staged:
            setattr(layer, name, param)
    for name, chunks in staged.items():
        first = chunks[0][2]
        assembled = torch.empty(
            (num_experts, *first.shape[1:]), dtype=first.dtype, device=device
        )
        for start, stop, pinned in chunks:
            assembled[start:stop].copy_(pinned, non_blocking=False)
        setattr(layer, name, torch.nn.Parameter(assembled, requires_grad=False))
    return True


def make_humming_deepep_input_schema(
    sublayer_name: str, shape_k: int
) -> HummingInputSchema:
    if shape_k % 128 != 0:
        raise ValueError(
            f"Humming FP8 dispatch requires {sublayer_name} K={shape_k} "
            "to be divisible by 128."
        )
    return HummingInputSchema(a_dtype="float8e4m3", input_scale_group_size=128)


def prepare_humming_layer(layer: LinearBase, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_schema = HummingInputSchema()

    shape_k_stacks = [layer.input_size_per_partition]
    shape_n_stacks = layer.output_partition_sizes

    # Step 1: convert weight to humming standard format
    weight_schema, tensors = weight_schema.convert_humming(
        tensors=layer.named_parameters(),
        shape_n_stacks=shape_n_stacks,
        shape_k_stacks=shape_k_stacks,
        param_dtype=layer.params_dtype,
    )

    layer.weight_schema = weight_schema

    for name, _ in list(layer.named_parameters()):
        delattr(layer, name)

    for name, tensor in tensors.items():
        param = torch.nn.Parameter(tensor, requires_grad=False)
        setattr(layer, name, param)

    # Step 2: transform weight (humming standard format) for forwarding
    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=layer.output_partition_sizes_sum,
        shape_k=layer.input_size_per_partition,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=256,
        pad_k_to_multiple=128,
        has_bias=layer.has_bias,
        torch_dtype=layer.param_dtype,
    )

    HummingMethod.transform_humming_layer(layer)


def prepare_humming_moe_layer(layer: FusedMoE, quant_config: dict):
    weight_schema = BaseWeightSchema.from_config(quant_config)
    input_quant_config = envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.get() or {}
    if humming_is_layer_skipped(input_quant_config, layer.layer_name):
        input_schema = HummingInputSchema()
    else:
        # TODO: read input_quant_config from quant_config
        input_schema = HummingInputSchema.from_config(input_quant_config)

    use_deepep_fp8_dispatch = configure_humming_deepep_dispatch(layer)

    shape_config = {
        "w13": (
            layer.intermediate_size_per_partition * 2,
            layer.hidden_size,
        ),
        "w2": (
            layer.hidden_size,
            layer.intermediate_size_per_partition,
        ),
    }

    layer.weight_schemas = {}
    layer.input_schemas = {}

    for sublayer_name in shape_config:
        # Step 1: convert weight to humming standard format
        tensors: dict[str, torch.Tensor] = dict(
            (key.removeprefix(sublayer_name + "_"), value)
            for key, value in layer.state_dict().items()
            if key.startswith(sublayer_name + "_")
        )

        shape_n, shape_k = shape_config[sublayer_name]
        shape_n_stacks = [shape_n]
        shape_k_stacks = [shape_k]
        if sublayer_name == "w13":
            shape_n_stacks = [shape_n // 2] * 2

        weight_schema_new, tensors = weight_schema.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            num_experts=layer.num_local_experts,
            param_dtype=layer.params_dtype,
        )

        layer.weight_schemas[sublayer_name] = weight_schema_new
        sub_input_schema = input_schema
        if use_deepep_fp8_dispatch:
            sub_input_schema = make_humming_deepep_input_schema(sublayer_name, shape_k)
        layer.input_schemas[sublayer_name] = sub_input_schema

        for name, _ in list(layer.named_parameters()):
            if not name.startswith(sublayer_name + "_"):
                continue
            delattr(layer, name)

        for name, tensor in tensors.items():
            name = f"{sublayer_name}_{name}"
            param = torch.nn.Parameter(tensor, requires_grad=False)
            setattr(layer, name, param)

        # Step 2: transform weight (humming standard format) for forwarding
        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            input_schema=sub_input_schema,
            weight_schema=weight_schema_new,
            has_bias=layer.with_bias,
            num_experts=layer.num_local_experts,
            torch_dtype=layer.params_dtype,
            sublayer_name=sublayer_name,
        )

        staged_ok = _transform_moe_sublayer_staged(
            layer,
            sublayer_name,
            meta_kwargs=dict(
                shape_n=shape_n,
                shape_k=shape_k,
                pad_n_to_multiple=256,
                pad_k_to_multiple=128,
                input_schema=sub_input_schema,
                weight_schema=weight_schema_new,
                has_bias=layer.with_bias,
                num_experts=layer.num_local_experts,
                torch_dtype=layer.params_dtype,
                sublayer_name=sublayer_name,
            ),
        )
        if not staged_ok:
            HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer_name)

    if not hasattr(layer, "locks"):
        device = layer.w13_weight.device
        locks = torch.zeros(1024, dtype=torch.int32, device=device)
        layer.register_buffer("locks", locks)
