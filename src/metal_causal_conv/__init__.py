"""
Metal-accelerated CausalConvWithState for Apple Silicon.

Fused depthwise causal convolution with persistent state carry and optional
SiLU activation. Drop-in replacement for the Conv1d + silu pattern used in
Mamba / Gated DeltaNet preprocessing.

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

    Args:
        input: Input tensor of shape (B, D, L).
        weight: Depthwise conv kernel of shape (D, 1, K).
        bias: Optional per-channel bias of shape (D,).
        past_state: Optional carry state from previous call, shape (B, D, K-1).
                    If None, treated as zeros.
        activation: "none" or "silu" (fused SiLU/Swish activation).

    Returns:
        output: Convolved output of shape (B, D, L).
        present_state: Updated carry state of shape (B, D, K-1) for next call.
    """
    if not _HAS_METAL:
        raise RuntimeError(
            "Metal extension not installed. Run: pip install -e . "
            "from the metal_causal_conv directory."
        )

    # Validate inputs
    assert input.ndim == 3, f"input must be 3D (B, D, L), got {input.ndim}D"
    assert weight.ndim == 3, f"weight must be 3D (D, 1, K), got {weight.ndim}D"
    assert weight.shape[1] == 1, f"weight dim 1 must be 1 (depthwise), got {weight.shape[1]}"

    B, D, L = input.shape
    K = weight.shape[2]
    state_len = K - 1

    assert weight.shape[0] == D, f"weight channels {weight.shape[0]} != input channels {D}"

    if bias is not None:
        assert bias.ndim == 1 and bias.shape[0] == D, \
            f"bias must be (D={D},), got {bias.shape}"

    if past_state is not None:
        assert past_state.shape == (B, D, state_len), \
            f"past_state must be ({B}, {D}, {state_len}), got {past_state.shape}"

    # Validate activation
    use_silu = activation in ("silu", "swish")
    if activation not in ("none", "silu", "swish", ""):
        raise ValueError(f"Unsupported activation: {activation!r}")

    # Ensure contiguous float32 inputs
    dtype_in = input.dtype
    input_f = input.float().contiguous()
    weight_f = weight.float().contiguous()

    bias_f = bias.float().contiguous() if bias is not None else torch.empty(0, device=input.device)
    state_f = past_state.float().contiguous() if past_state is not None else torch.empty(0, device=input.device)

    has_bias = bias is not None
    has_state = past_state is not None

    # Call Metal kernel
    output, present_state = causal_conv_metal_cpp.causal_conv_with_state_fwd(
        input_f,
        weight_f,
        bias_f,
        state_f,
        has_bias,
        has_state,
        use_silu,
        _SHADER_PATH,
    )

    # Restore original dtype
    output = output.to(dtype=dtype_in)
    present_state = present_state.to(dtype=dtype_in)

    return output, present_state


__all__ = ['causal_conv_with_state']
