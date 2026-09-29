# SPDX-License-Identifier: Apache-2.0
"""Metal CausalConvWithState reference kernel for Apple Silicon.

Fused depthwise causal convolution with persistent state carry and optional
SiLU activation for inference with PyTorch/MPS.

Inputs:
    input:      (B, D, L) — input tensor
    weight:     (D, 1, K) — depthwise kernel weights
    bias:       (D,) or None — optional per-channel bias
    past_state: (B, D, K-1) or None — carry state from previous chunk
    activation: "none" or "silu" — optional fused activation

Outputs:
    output:        (B, D, L)   — convolved (+ activated) output
    present_state: (B, D, K-1) — carry state for next chunk
"""
import os
import torch

try:
    import causal_conv_metal_cpp
    _HAS_METAL = True
except ImportError:
    _HAS_METAL = False

_SHADER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'causal_conv.metal')


def causal_conv_with_state(
    input: torch.Tensor,             # (B, D, L)
    weight: torch.Tensor,            # (D, 1, K)
    bias: torch.Tensor = None,       # (D,) or None
    past_state: torch.Tensor = None, # (B, D, K-1) or None
    activation: str = "none",        # "none" or "silu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused causal depthwise convolution with state carry on Metal.
    """
    if not _HAS_METAL:
        raise RuntimeError("Metal extension not installed.")

    # Validate inputs
    assert input.ndim == 3, f"input must be 3D (B, D, L), got {input.ndim}D"
    assert weight.ndim == 3, f"weight must be 3D (D, 1, K), got {weight.ndim}D"
    assert weight.shape[1] == 1, f"weight dim 1 must be 1 (depthwise), got {weight.shape[1]}"

    B, D, L = input.shape
    K = weight.shape[2]
    state_len = K - 1

    assert weight.shape[0] == D, f"weight channels {weight.shape[0]} != input channels {D}"
    assert input.dtype in (torch.float32, torch.float16), "Only float32 and float16 supported"

    # Validate activation
    use_silu = activation in ("silu", "swish")
    if activation not in ("none", "silu", "swish", ""):
        raise ValueError(f"Unsupported activation: {activation!r}")

    # Ensure contiguous inputs. Only cast if types don't match.
    # Weight and bias should match input dtype.
    input_c = input.contiguous()
    weight_c = weight.to(dtype=input.dtype).contiguous()
    
    bias_c = bias.to(dtype=input.dtype).contiguous() if bias is not None else torch.empty(0, device=input.device, dtype=input.dtype)
    state_c = past_state.contiguous() if past_state is not None else torch.empty(0, device=input.device, dtype=input.dtype)

    has_bias = bias is not None
    has_state = past_state is not None

    # Call Metal kernel (handles float and half internally)
    output, present_state = causal_conv_metal_cpp.causal_conv_with_state_fwd(
        input_c,
        weight_c,
        bias_c,
        state_c,
        has_bias,
        has_state,
        use_silu,
        _SHADER_PATH,
    )

    return output, present_state


__all__ = ['causal_conv_with_state']
