"""Compatibility helpers for branching Hugging Face DynamicCache objects.

This module deliberately imports neither torch nor transformers so both the
legacy and newer cache API shapes can be regression-tested in the lightweight
CI job.
"""

from __future__ import annotations


def layer_tensor_pairs(cache):
    """Return existing per-layer (K, V) tensors from the newer cache API."""

    layers = getattr(cache, "layers", None)
    if layers is None:
        return None

    pairs = []
    for index, layer in enumerate(layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is None or values is None:
            raise RuntimeError(
                f"DynamicCache layer {index} exposes no keys/values tensors"
            )
        pairs.append((keys, values))
    return pairs


def cache_fork_mode(cache, dynamic_cache_cls) -> str:
    """Return the supported reconstruction path for a cache/class pair."""

    to_legacy = getattr(cache, "to_legacy_cache", None)
    from_legacy = getattr(dynamic_cache_cls, "from_legacy_cache", None)
    if callable(to_legacy) and callable(from_legacy):
        return "legacy-conversion"
    if layer_tensor_pairs(cache) is not None:
        return "layer-tensor-constructor"
    return "unsupported"


def fork_dynamic_cache(cache, dynamic_cache_cls, *, version="unknown"):
    """Create an independent cache container sharing historical K/V read-only.

    Older Transformers exposes legacy conversion helpers. Newer Transformers
    removed those helpers and reconstructs DynamicCache from existing per-layer
    K/V using the ddp_cache_data constructor input.
    """

    mode = cache_fork_mode(cache, dynamic_cache_cls)

    if mode == "legacy-conversion":
        return (
            dynamic_cache_cls.from_legacy_cache(cache.to_legacy_cache()),
            mode,
        )

    if mode == "layer-tensor-constructor":
        pairs = layer_tensor_pairs(cache)
        try:
            branch = dynamic_cache_cls(ddp_cache_data=pairs)
        except TypeError as keyword_error:
            try:
                branch = dynamic_cache_cls(pairs)
            except Exception as positional_error:
                raise RuntimeError(
                    "could not reconstruct DynamicCache branch under "
                    f"transformers {version}; "
                    f"keyword error={keyword_error!r}; "
                    f"positional error={positional_error!r}"
                ) from positional_error
        return branch, mode

    raise RuntimeError(
        "unsupported DynamicCache API in transformers "
        f"{version}: neither legacy conversion nor cache.layers K/V tensors "
        "are available"
    )
