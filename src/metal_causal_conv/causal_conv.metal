/******************************************************************************
 * Metal CausalConvWithState — Flat 1D High-Occupancy Kernel
 *
 * Eliminates shared memory and threadgroup barriers entirely.
 * Each thread computes ONE output element from a flat thread index
 * over the (B × D × L) output space. Apple Silicon L1/L2 cache
 * handles the small K-element overlap naturally (K is typically 3–4).
 *
 * This maximizes GPU occupancy by allowing the driver full freedom
 * in threadgroup packing — no per-channel threadgroup partitioning.
 *
 * Copyright 2026. MIT License.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

struct CausalConvParams {
    uint batch_size;
    uint channels;
    uint input_length;
    uint kernel_size;
    uint state_length;
    uint has_bias;
    uint has_conv_state;
    uint use_silu;
};

template<typename T>
inline T silu(T x) {
    return x / (1.0f + exp(-x));
}

template<typename T>
void causal_conv_flat_impl(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device const T* conv_state,
    device T* output,
    device T* present_state,
    constant CausalConvParams& params,
    uint tid)
{
    const uint L = params.input_length;
    const uint D = params.channels;
    const uint K = params.kernel_size;
    const uint state_len = params.state_length;
    const uint total = params.batch_size * D * L;

    if (tid >= total) return;

    // Decode flat index → (batch, channel, position)
    const uint l = tid % L;
    const uint d = (tid / L) % D;
    const uint b = tid / (L * D);

    const uint data_base  = (b * D + d) * L;
    const uint state_base = (b * D + d) * state_len;
    const uint weight_base = d * K;

    // Convolution: dot product of K elements
    float acc = 0.0f;
    for (uint j = 0; j < K; j++) {
        int src_pos = (int)l - (int)state_len + (int)j;
        float val;
        if (src_pos < 0) {
            // Read from convolution state (past chunk's tail)
            val = params.has_conv_state
                ? (float)conv_state[state_base + (uint)((int)state_len + src_pos)]
                : 0.0f;
        } else {
            val = (float)input[data_base + (uint)src_pos];
        }
        acc += val * (float)weight[weight_base + j];
    }

    if (params.has_bias) {
        acc += (float)bias[d];
    }

    if (params.use_silu) {
        acc = (float)silu(acc);
    }

    output[data_base + l] = (T)acc;

    // State update: last position in the sequence writes present_state
    if (l == L - 1) {
        for (uint j = 0; j < state_len; j++) {
            int src = (int)l - (int)state_len + 1 + (int)j;
            float s;
            if (src < 0) {
                s = params.has_conv_state
                    ? (float)conv_state[state_base + (uint)((int)state_len + src)]
                    : 0.0f;
            } else {
                s = (float)input[data_base + (uint)src];
            }
            present_state[state_base + j] = (T)s;
        }
    }
}

// Explicit instantiations
kernel void causal_conv_with_state_fwd_float(
    device const float* input         [[buffer(0)]],
    device const float* weight        [[buffer(1)]],
    device const float* bias          [[buffer(2)]],
    device const float* conv_state    [[buffer(3)]],
    device float* output              [[buffer(4)]],
    device float* present_state       [[buffer(5)]],
    constant CausalConvParams& params [[buffer(6)]],
    uint tid                          [[thread_position_in_grid]])
{
    causal_conv_flat_impl<float>(input, weight, bias, conv_state, output, present_state, params, tid);
}

kernel void causal_conv_with_state_fwd_half(
    device const half* input          [[buffer(0)]],
    device const half* weight         [[buffer(1)]],
    device const half* bias           [[buffer(2)]],
    device const half* conv_state     [[buffer(3)]],
    device half* output               [[buffer(4)]],
    device half* present_state        [[buffer(5)]],
    constant CausalConvParams& params [[buffer(6)]],
    uint tid                          [[thread_position_in_grid]])
{
    causal_conv_flat_impl<half>(input, weight, bias, conv_state, output, present_state, params, tid);
}