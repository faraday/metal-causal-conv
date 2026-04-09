/******************************************************************************
 * Metal CausalConvWithState — Fused Depthwise Causal Conv + SiLU
 *
 * Metal compute shader for stateful causal depthwise 1D convolution with
 * optional fused SiLU activation.
 *
 * Inputs:
 *   input:      (B, D, L)   — input tensor
 *   weight:     (D, 1, K)   — depthwise kernel (stored as D*K flat)
 *   bias:       (D,)        — optional per-channel bias
 *   conv_state: (B, D, K-1) — optional carry state from previous chunk
 *
 * Outputs:
 *   output:        (B, D, L)   — convolved + activated output
 *   present_state: (B, D, K-1) — updated carry state
 *
 * Grid: one thread per (batch, channel, position) triple.
 *
 * The virtual input is conceptually [conv_state | input], giving a total
 * length of (K-1) + L. For output position `pos`, we compute:
 *
 *   output[b,d,pos] = sum_{j=0}^{K-1} weight[d,j] * virtual[pos+j]
 *
 * where virtual[i] = conv_state[b,d,i] for i < K-1
 *                   = input[b,d,i-(K-1)]  for i >= K-1
 *
 * The thread at pos==0 also writes present_state = last K-1 of virtual input.
 *
 * Copyright 2026. MIT License.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void causal_conv_with_state_fwd(
    device const float* input       [[buffer(0)]],   // (B, D, L)
    device const float* weight      [[buffer(1)]],   // (D, K) flattened from (D, 1, K)
    device const float* bias        [[buffer(2)]],   // (D,)
    device const float* conv_state  [[buffer(3)]],   // (B, D, state_len)
    device float* output            [[buffer(4)]],   // (B, D, L)
    device float* present_state     [[buffer(5)]],   // (B, D, state_len)
    constant uint& batch_size       [[buffer(6)]],
    constant uint& channels         [[buffer(7)]],
    constant uint& input_length     [[buffer(8)]],
    constant uint& kernel_size      [[buffer(9)]],
    constant uint& state_length     [[buffer(10)]],
    constant uint& output_size      [[buffer(11)]],
    constant uint& has_bias         [[buffer(12)]],
    constant uint& has_conv_state   [[buffer(13)]],
    constant uint& use_silu         [[buffer(14)]],
    uint tid                        [[thread_position_in_grid]])
{
    if (tid >= output_size) return;

    // Decompose linear index -> (batch, channel, position)
    uint pos = tid % input_length;
    uint bc_idx = tid / input_length;
    uint channel_idx = bc_idx % channels;
    uint batch_idx = bc_idx / channels;

    // Accumulate depthwise convolution
    float acc = 0.0f;
    uint weight_base = channel_idx * kernel_size;

    for (uint j = 0; j < kernel_size; j++) {
        uint virtual_pos = pos + j;
        float val = 0.0f;

        if (has_conv_state) {
            if (virtual_pos < state_length) {
                // Read from conv_state: (B, D, state_length)
                uint state_idx = (batch_idx * channels + channel_idx) * state_length + virtual_pos;
                val = conv_state[state_idx];
            } else {
                // Read from input: (B, D, L)
                uint input_pos = virtual_pos - state_length;
                uint input_idx = (batch_idx * channels + channel_idx) * input_length + input_pos;
                val = input[input_idx];
            }
        } else {
            // No state: zero-pad for positions before input starts
            if (virtual_pos >= state_length) {
                uint input_pos = virtual_pos - state_length;
                uint input_idx = (batch_idx * channels + channel_idx) * input_length + input_pos;
                val = input[input_idx];
            }
            // else val stays 0 (zero padding)
        }

        float w = weight[weight_base + j];
        acc += val * w;
    }

    // Add bias
    if (has_bias) {
        acc += bias[channel_idx];
    }

    // Apply SiLU activation
    if (use_silu) {
        acc = silu(acc);
    }

    // Write output: (B, D, L)
    uint out_idx = (batch_idx * channels + channel_idx) * input_length + pos;
    output[out_idx] = acc;

    // Write present_state: last (K-1) elements of virtual input.
    // Only the thread at pos==0 for each (batch, channel) handles this.
    if (pos == 0u) {
        for (uint s = 0; s < state_length; s++) {
            float state_val = 0.0f;
            // We want the last state_length elements of virtual input.
            // Virtual input total length = state_length + input_length.
            // Last state_length elements start at index input_length.
            uint vp = input_length + s;

            if (has_conv_state) {
                if (vp < state_length) {
                    uint si = (batch_idx * channels + channel_idx) * state_length + vp;
                    state_val = conv_state[si];
                } else {
                    uint ip = vp - state_length;
                    uint ii = (batch_idx * channels + channel_idx) * input_length + ip;
                    state_val = input[ii];
                }
            } else {
                if (vp >= state_length) {
                    uint ip = vp - state_length;
                    uint ii = (batch_idx * channels + channel_idx) * input_length + ip;
                    state_val = input[ii];
                }
            }

            uint ps_idx = (batch_idx * channels + channel_idx) * state_length + s;
            present_state[ps_idx] = state_val;
        }
    }
}
