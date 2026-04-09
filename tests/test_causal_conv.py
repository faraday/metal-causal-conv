"""
Tests for metal_causal_conv — CausalConvWithState + fused SiLU on Metal.

These tests validate the Metal kernel against a pure-PyTorch reference
implementation. The reference performs the exact same computation:
  1. Concatenate [past_state, input] along the time axis
  2. Depthwise conv1d (groups=channels) with kernel_size=K
  3. Slice to output length L (causal — no future leakage)
  4. Optionally add bias
  5. Optionally apply SiLU activation
  6. Extract present_state = last K-1 samples of virtual input

Tests cover:
  - Basic forward pass (no state, no bias, no activation)
  - Forward with bias
  - Forward with fused SiLU activation
  - Stateful streaming: carry state across multiple chunks
  - Chunk boundary correctness vs. single-pass processing
  - Edge cases: L=1 (single sample), various batch sizes
  - dtype preservation (float32, float16)
"""

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation
# ---------------------------------------------------------------------------

def causal_conv_with_state_ref(
    input: torch.Tensor,        # (B, D, L)
    weight: torch.Tensor,       # (D, 1, K)
    bias: torch.Tensor = None,  # (D,) or None
    past_state: torch.Tensor = None,  # (B, D, K-1) or None
    activation: str = "none",   # "none" or "silu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference CausalConvWithState in pure PyTorch.

    Returns:
        output: (B, D, L)
        present_state: (B, D, K-1)
    """
    B, D, L = input.shape
    K = weight.shape[2]
    state_len = K - 1

    # Build virtual input: [past_state | input]
    if past_state is not None:
        virtual_input = torch.cat([past_state, input], dim=2)  # (B, D, state_len + L)
    else:
        virtual_input = torch.cat([
            torch.zeros(B, D, state_len, device=input.device, dtype=input.dtype),
            input
        ], dim=2)  # (B, D, state_len + L)

    # Depthwise conv1d with no padding (valid mode)
    # groups=D makes it depthwise
    output = F.conv1d(virtual_input, weight, bias=None, groups=D)
    # output shape: (B, D, state_len + L - K + 1) = (B, D, L)

    # Add bias
    if bias is not None:
        output = output + bias.unsqueeze(0).unsqueeze(2)  # (1, D, 1)

    # Apply activation
    if activation == "silu" or activation == "swish":
        output = F.silu(output)

    # Extract present_state: last K-1 samples of virtual_input
    present_state = virtual_input[:, :, -state_len:]  # (B, D, K-1)

    return output, present_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture
def dims():
    """Default test dimensions matching speech enhancement use case."""
    return {
        "B": 1,       # batch size
        "D": 768,     # d_inner (channels)
        "L": 256,     # chunk length
        "K": 4,       # kernel size (d_conv)
    }


@pytest.fixture
def small_dims():
    """Small dimensions for quick validation."""
    return {
        "B": 1,
        "D": 4,
        "L": 8,
        "K": 4,
    }


def make_inputs(B, D, L, K, device, dtype=torch.float32):
    """Create random test inputs."""
    torch.manual_seed(42)
    input_t = torch.randn(B, D, L, device=device, dtype=dtype)
    weight = torch.randn(D, 1, K, device=device, dtype=dtype) * 0.1
    bias = torch.randn(D, device=device, dtype=dtype) * 0.01
    past_state = torch.randn(B, D, K - 1, device=device, dtype=dtype) * 0.5
    return input_t, weight, bias, past_state


# ---------------------------------------------------------------------------
# Tests: Reference Implementation (validates the ref itself)
# ---------------------------------------------------------------------------

class TestReference:
    """Validate the reference implementation before comparing Metal to it."""

    def test_output_shape(self, device, small_dims):
        d = small_dims
        inp, weight, bias, state = make_inputs(**d, device=device)
        out, present = causal_conv_with_state_ref(inp, weight)
        assert out.shape == (d["B"], d["D"], d["L"])
        assert present.shape == (d["B"], d["D"], d["K"] - 1)

    def test_no_state_is_zero_padded(self, device, small_dims):
        d = small_dims
        inp, weight, _, _ = make_inputs(**d, device=device)

        # Explicit zeros vs. None should give same result
        zero_state = torch.zeros(d["B"], d["D"], d["K"] - 1, device=device)
        out_none, _ = causal_conv_with_state_ref(inp, weight, past_state=None)
        out_zero, _ = causal_conv_with_state_ref(inp, weight, past_state=zero_state)
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    def test_causality(self, device, small_dims):
        """Changing a future input sample should not affect earlier outputs."""
        d = small_dims
        inp, weight, _, _ = make_inputs(**d, device=device)

        out1, _ = causal_conv_with_state_ref(inp, weight)

        inp2 = inp.clone()
        inp2[:, :, -1] += 100.0  # perturb last sample
        out2, _ = causal_conv_with_state_ref(inp2, weight)

        # All positions except last K-1 should be identical
        # Actually, only the last position output should change (before activation)
        # because we changed inp[..., L-1] which only affects out[..., L-1] through
        # the causal window
        assert torch.allclose(out1[:, :, :-1], out2[:, :, :-1], atol=1e-6)

    def test_streaming_equivalence(self, device, small_dims):
        """Processing in 2 chunks with state carry = processing full sequence at once."""
        d = small_dims
        B, D, L, K = d["B"], d["D"], d["L"], d["K"]

        torch.manual_seed(42)
        full_input = torch.randn(B, D, L, device=device)
        weight = torch.randn(D, 1, K, device=device) * 0.1
        bias = torch.randn(D, device=device) * 0.01

        # Single pass
        out_full, _ = causal_conv_with_state_ref(full_input, weight, bias=bias)

        # Two chunks with state carry
        L_half = L // 2
        chunk1 = full_input[:, :, :L_half]
        chunk2 = full_input[:, :, L_half:]

        out1, state1 = causal_conv_with_state_ref(chunk1, weight, bias=bias, past_state=None)
        out2, state2 = causal_conv_with_state_ref(chunk2, weight, bias=bias, past_state=state1)

        out_chunked = torch.cat([out1, out2], dim=2)

        assert torch.allclose(out_full, out_chunked, atol=1e-5), \
            f"Max diff: {(out_full - out_chunked).abs().max().item()}"

    def test_silu_activation(self, device, small_dims):
        """SiLU activation should match F.silu applied after conv."""
        d = small_dims
        inp, weight, bias, state = make_inputs(**d, device=device)

        # Manual: conv then silu
        out_no_act, present = causal_conv_with_state_ref(inp, weight, bias=bias,
                                                          past_state=state, activation="none")
        expected = F.silu(out_no_act)

        # Fused
        out_silu, present2 = causal_conv_with_state_ref(inp, weight, bias=bias,
                                                         past_state=state, activation="silu")

        assert torch.allclose(expected, out_silu, atol=1e-6)
        assert torch.allclose(present, present2, atol=1e-6)  # state should be same


# ---------------------------------------------------------------------------
# Tests: Metal kernel vs Reference
# ---------------------------------------------------------------------------

class TestMetalKernel:
    """Compare Metal kernel output against PyTorch reference."""

    @pytest.fixture(autouse=True)
    def _load_metal(self):
        try:
            from metal_causal_conv import causal_conv_with_state
            self.metal_fn = causal_conv_with_state
        except ImportError:
            pytest.skip("metal_causal_conv not installed (pip install -e .)")

    def test_basic_no_state_no_bias(self, device, dims):
        d = dims
        inp, weight, _, _ = make_inputs(**d, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(inp, weight)
        out_metal, state_metal = self.metal_fn(inp, weight)

        assert out_metal.shape == out_ref.shape
        assert state_metal.shape == state_ref.shape
        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Output max diff: {(out_ref - out_metal).abs().max().item()}"
        assert torch.allclose(state_ref, state_metal, atol=1e-4), \
            f"State max diff: {(state_ref - state_metal).abs().max().item()}"

    def test_with_bias(self, device, dims):
        d = dims
        inp, weight, bias, _ = make_inputs(**d, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(inp, weight, bias=bias)
        out_metal, state_metal = self.metal_fn(inp, weight, bias=bias)

        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Output max diff: {(out_ref - out_metal).abs().max().item()}"

    def test_with_silu(self, device, dims):
        d = dims
        inp, weight, bias, state = make_inputs(**d, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(
            inp, weight, bias=bias, past_state=state, activation="silu")
        out_metal, state_metal = self.metal_fn(
            inp, weight, bias=bias, past_state=state, activation="silu")

        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Output max diff: {(out_ref - out_metal).abs().max().item()}"
        assert torch.allclose(state_ref, state_metal, atol=1e-4), \
            f"State max diff: {(state_ref - state_metal).abs().max().item()}"

    def test_with_state(self, device, dims):
        d = dims
        inp, weight, bias, state = make_inputs(**d, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(
            inp, weight, bias=bias, past_state=state)
        out_metal, state_metal = self.metal_fn(
            inp, weight, bias=bias, past_state=state)

        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Output max diff: {(out_ref - out_metal).abs().max().item()}"
        assert torch.allclose(state_ref, state_metal, atol=1e-4), \
            f"State max diff: {(state_ref - state_metal).abs().max().item()}"

    def test_streaming_two_chunks(self, device, dims):
        """Metal kernel produces correct results when streaming across chunks."""
        d = dims
        B, D, L, K = d["B"], d["D"], d["L"], d["K"]

        torch.manual_seed(123)
        full_input = torch.randn(B, D, L, device=device)
        weight = torch.randn(D, 1, K, device=device) * 0.1
        bias = torch.randn(D, device=device) * 0.01

        # Single pass reference
        out_full, _ = causal_conv_with_state_ref(full_input, weight, bias=bias)

        # Two-chunk streaming via Metal
        L_half = L // 2
        out1, state1 = self.metal_fn(
            full_input[:, :, :L_half], weight, bias=bias, past_state=None)
        out2, state2 = self.metal_fn(
            full_input[:, :, L_half:], weight, bias=bias, past_state=state1)

        out_streamed = torch.cat([out1, out2], dim=2)

        assert torch.allclose(out_full, out_streamed, atol=1e-4), \
            f"Streaming max diff: {(out_full - out_streamed).abs().max().item()}"

    def test_streaming_many_chunks(self, device):
        """Stream 8 small chunks and verify against single-pass reference."""
        B, D, K = 1, 128, 4
        chunk_size = 32
        num_chunks = 8
        L = chunk_size * num_chunks

        torch.manual_seed(456)
        full_input = torch.randn(B, D, L, device=device)
        weight = torch.randn(D, 1, K, device=device) * 0.1

        # Reference single pass
        out_ref, _ = causal_conv_with_state_ref(full_input, weight)

        # Chunked Metal
        state = None
        chunks_out = []
        for i in range(num_chunks):
            chunk = full_input[:, :, i * chunk_size:(i + 1) * chunk_size]
            out_chunk, state = self.metal_fn(chunk, weight, past_state=state)
            chunks_out.append(out_chunk)

        out_streamed = torch.cat(chunks_out, dim=2)

        assert torch.allclose(out_ref, out_streamed, atol=1e-4), \
            f"Multi-chunk streaming max diff: {(out_ref - out_streamed).abs().max().item()}"

    def test_single_sample(self, device):
        """L=1: single sample per chunk (decode mode)."""
        B, D, K = 1, 64, 4

        torch.manual_seed(789)
        weight = torch.randn(D, 1, K, device=device) * 0.1
        bias = torch.randn(D, device=device) * 0.01

        state = None
        for _ in range(10):
            inp = torch.randn(B, D, 1, device=device)
            out_ref, state_ref = causal_conv_with_state_ref(
                inp, weight, bias=bias, past_state=state, activation="silu")
            out_metal, state_metal = self.metal_fn(
                inp, weight, bias=bias, past_state=state, activation="silu")

            assert torch.allclose(out_ref, out_metal, atol=1e-4), \
                f"Decode max diff: {(out_ref - out_metal).abs().max().item()}"
            assert torch.allclose(state_ref, state_metal, atol=1e-4)

            state = state_metal  # carry forward Metal state

    def test_batch_size(self, device):
        """Multi-batch correctness."""
        B, D, L, K = 4, 128, 64, 4
        inp, weight, bias, state = make_inputs(B, D, L, K, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(
            inp, weight, bias=bias, past_state=state, activation="silu")
        out_metal, state_metal = self.metal_fn(
            inp, weight, bias=bias, past_state=state, activation="silu")

        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Batch max diff: {(out_ref - out_metal).abs().max().item()}"

    def test_large_channel_count(self, device):
        """Test with d_inner=1536 (Mamba-sized)."""
        B, D, L, K = 1, 1536, 128, 4
        inp, weight, bias, state = make_inputs(B, D, L, K, device=device)

        out_ref, state_ref = causal_conv_with_state_ref(
            inp, weight, bias=bias, past_state=state)
        out_metal, state_metal = self.metal_fn(
            inp, weight, bias=bias, past_state=state)

        assert torch.allclose(out_ref, out_metal, atol=1e-4), \
            f"Large channel max diff: {(out_ref - out_metal).abs().max().item()}"

    def test_float16(self, device):
        """Verify float16 support (reduced tolerance)."""
        B, D, L, K = 1, 128, 64, 4
        inp, weight, bias, state = make_inputs(B, D, L, K, device=device, dtype=torch.float16)

        out_ref, state_ref = causal_conv_with_state_ref(
            inp, weight, bias=bias, past_state=state, activation="silu")
        out_metal, state_metal = self.metal_fn(
            inp, weight, bias=bias, past_state=state, activation="silu")

        assert torch.allclose(out_ref, out_metal, atol=1e-2, rtol=1e-2), \
            f"FP16 max diff: {(out_ref - out_metal).abs().max().item()}"


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
