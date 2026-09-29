# Metal Causal Conv

Metal Causal Conv is a small, experimental, inference-only PyTorch reference implementation of stateful causal depthwise 1D convolution for Apple Silicon. It computes the convolution and optional SiLU activation in a native Metal kernel and returns the state needed to continue on the next chunk.

This is a restricted reference implementation, not a general-purpose convolution package. It was used with PyTorch on an Apple M1 Max.

## Requirements and installation

- Apple Silicon Mac with a working PyTorch MPS backend
- Python 3.10 or newer
- PyTorch 2.0 or newer with MPS support
- Xcode Command Line Tools and the macOS Metal SDK

Install PyTorch and the build tools in the environment you will use to run the kernel. Then build the extension without pip's isolated build environment so it uses that environment's PyTorch headers:

```bash
git clone https://github.com/faraday/metal-causal-conv.git
cd metal-causal-conv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch setuptools wheel ninja
python -m pip install --no-build-isolation -e .
```

The native extension can be built when MPS is unavailable, but execution requires an Apple Silicon machine with a working MPS backend. There is no CPU fallback.

## Streaming example

```python
import torch
from metal_causal_conv import causal_conv_with_state

device = "mps"
batch, channels, chunk_length, kernel_size = 1, 64, 32, 4
weight = torch.randn(channels, 1, kernel_size, device=device) * 0.1
bias = torch.randn(channels, device=device) * 0.01
first_chunk = torch.randn(batch, channels, chunk_length, device=device)
second_chunk = torch.randn(batch, channels, chunk_length, device=device)

with torch.inference_mode():
    first_output, state = causal_conv_with_state(
        first_chunk, weight, bias=bias, activation="silu"
    )
    second_output, state = causal_conv_with_state(
        second_chunk, weight, bias=bias, past_state=state, activation="silu"
    )

assert first_output.shape == (batch, channels, chunk_length)
assert second_output.shape == first_output.shape
assert state.shape == (batch, channels, kernel_size - 1)
```

Each call returns `(output, present_state)`. Pass `present_state` as `past_state` on the next call. If no state is supplied, the kernel uses zero history. State holds the most recent `K-1` input samples in oldest-to-newest order; the returned state is independent of the selected activation.

## Input and output contract

| Argument | Shape | Required | Notes |
| --- | --- | --- | --- |
| `input` | `[B, D, L]` | Yes | MPS tensor; `B`, `D`, and `L` must be positive. |
| `weight` | `[D, 1, K]` | Yes | Depthwise kernel; `K` must be at least 2. |
| `bias` | `[D]` | No | Added before the activation. |
| `past_state` | `[B, D, K-1]` | No | MPS tensor with the same dtype as `input`. |
| `activation` | string | No | `"none"` (default), `"silu"`, or alias `"swish"`; `""` also means no activation. |

All provided tensors must already be on MPS. The wrapper accepts float32 and float16 input. It converts `weight` and `bias` to the input dtype; `past_state` must already have that dtype. The kernel accumulates in float32 and casts output and state back to the input dtype.

The wrapper has limited runtime validation. Callers must follow the documented shape, device, and dtype contract; invalid inputs may fail during dispatch. The extension has no autograd implementation and is intended for inference under `torch.inference_mode()`.

## Related work

[Metal SSM](https://github.com/faraday/metal-ssm) is a companion PyTorch/MPS reference kernel for the Mamba selective-scan recurrence. [SpeechLens](https://github.com/faraday/SpeechLens) is a related Apple Silicon project with a separate Swift/MLX implementation.

## License

Metal Causal Conv is licensed under the [Apache License 2.0](LICENSE).
