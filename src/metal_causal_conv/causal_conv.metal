/******************************************************************************
 * Metal CausalConvWithState — Fused Depthwise Causal Conv + SiLU
 *
 * Metal compute shader for stateful causal depthwise 1D convolution with
 * optional fused SiLU activation.
 *
 * Inputs:
 * input:      (B, D, L)   — input tensor
 * weight:     (D, 1, K)   — depthwise kernel (stored as D*K flat)
 * bias:       (D,)        — optional per-channel bias
 * conv_state: (B, D, K-1) — optional carry state from previous chunk
 *
 * Outputs:
 * output:        (B, D, L)   — convolved + activated output
 * present_state: (B, D, K-1) — updated carry state
 *
 * Grid: 3D Grid (L, D, B). tid.x = pos, tid.y = channel, tid.z = batch.
 *
 * Copyright 2026. MIT License.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Struct to pass all scalar parameters in a single API call (reduces CPU overhead)
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

// SiLU (Swish): x * sigmoid(x) = x / (1 + exp(-x))
inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void causal_conv_with_state_fwd(
    device const float* input         [[buffer(0)]],   // (B, D, L)
    constant float* weight            [[buffer(1)]],   // (D, K) - constant cache
    constant float* bias              [[buffer(2)]],   // (D,) - constant cache
    device const float* conv_state    [[buffer(3)]],   // (B, D, state_len)
    device float* output              [[buffer(4)]],   // (B, D, L)
    device float* present_state       [[buffer(5)]],   // (B, D, state_len)
    constant CausalConvParams& params [[buffer(6)]],
    uint3 tid                         [[thread_position_in_grid]])
{
    // Extract coordinates directly from the 3D grid (zero integer division)
    uint pos = tid.x;
    uint channel_idx = tid.y;
    uint batch_idx = tid.z;

    // Bounds check
    if (pos >= params.input_length || channel_idx >= params.channels || batch_idx >= params.batch_size) {
        return;
    }

    uint kernel_size = params.kernel_size;
    uint state_length = params.state_length;

    // Hoist base index calculations out of the loops
    uint weight_base = channel_idx * kernel_size;
    uint state_base_idx = (batch_idx * params.channels + channel_idx) * state_length;
    uint input_base_idx = (batch_idx * params.channels + channel_idx) * params.input_length;

    float acc = 0.0f;

    // Loop Splitting: Find the exact index where we stop reading state and start reading input
    // virtual_pos = pos + j. Boundary is when pos + j == state_length => j == state_length - pos
    uint split_j = (pos < state_length) ? (state_length - pos) : 0;
    split_j = min(split_j, kernel_size);

    // Part A: Read from conv_state (or zero pad if no state exists)
    if (split_j > 0) {
        if (params.has_conv_state) {
            for (uint j = 0; j < split_j; j++) {
                acc += conv_state[state_base_idx + pos + j] * weight[weight_base + j];
            }
        } 
        // If has_conv_state is 0, these values represent padding before the sequence,
        // so acc += 0 * w. We can completely skip the loop.
    }

    // Part B: Read directly from the input tensor (completely branchless)
    for (uint j = split_j; j < kernel_size; j++) {
        uint input_pos = (pos + j) - state_length;
        acc += input[input_base_idx + input_pos] * weight[weight_base + j];
    }

    // Add optional bias
    if (params.has_bias) {
        acc += bias[channel_idx];
    }

    // Apply optional SiLU activation
    if (params.use_silu) {
        acc = silu(acc);
    }

    // Write final convolved output
    output[input_base_idx + pos] = acc;

    // Write present_state: last (K-1) elements of the virtual input.
    // Handled by the first thread of each sequence to prevent race conditions.
    if (pos == 0u) {
        for (uint s = 0; s < state_length; s++) {
            uint vp = params.input_length + s;
            float state_val = 0.0f;
            
            if (vp < state_length) {
                if (params.has_conv_state) {
                    state_val = conv_state[state_base_idx + vp];
                }
            } else {
                state_val = input[input_base_idx + vp - state_length];
            }
            
            present_state[state_base_idx + s] = state_val;
        }
    }
}