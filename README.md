# metal_causal_conv

Fused **CausalConvWithState + SiLU** Metal kernel for Apple Silicon (MPS).

A stateful causal depthwise 1D convolution that persists the last `d_conv-1` samples across chunks, fused with optional SiLU activation — all in a single Metal kernel dispatch.

## Install

```bash
pip install -e .
```

## Usage

```python
from metal_causal_conv import causal_conv_with_state

# First chunk
output, state = causal_conv_with_state(
    input,      # (B, D, L) on MPS
    weight,     # (D, 1, K)
    bias=bias,  # (D,) optional
    activation="silu",
)

# Subsequent chunks — carry state forward
output, state = causal_conv_with_state(
    next_input, weight, bias=bias,
    past_state=state,
    activation="silu",
)
```

## Test

```bash
pytest tests/ -v
```
