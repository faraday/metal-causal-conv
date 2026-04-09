/******************************************************************************
 * Metal CausalConvWithState — Optimized Fused Depthwise Causal Conv + SiLU
 *
 * One thread per (Batch, Channel) pair. Each thread processes the entire 
 * sequence L, maintaining the convolution sliding window in registers.
 *
 * This architecture maximizes memory coalescing (sequential threads process 
 * sequential channels) and eliminates the overhead of launching one thread 
 * per output element.
 *
 * Copyright 2026. MIT License.
 ******************************************************************************/

#include <metal_stdlib>
using namespace metal;

// Struct to pass all scalar parameters
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

// SiLU (Swish): x * sigmoid(x)
template<typename T>
inline T silu(T x) {
    return x / (1.0f + exp(-x));
}

template<typename T>
kernel void causal_conv_with_state_kernel(
    device const T* input             [[buffer(0)]],   // (B, D, L)
    device const T* weight            [[buffer(1)]],   // (D, K)
    device const T* bias              [[buffer(2)]],   // (D,)
    device const T* conv_state        [[buffer(3)]],   // (B, D, K-1)
    device T* output                  [[buffer(4)]],   // (B, D, L)
    device T* present_state           [[buffer(5)]],   // (B, D, K-1)
    constant CausalConvParams& params [[buffer(6)]],
    uint2 tid                         [[thread_position_in_grid]])
{
    uint channel_idx = tid.x;
    uint batch_idx = tid.y;

    if (channel_idx >= params.channels || batch_idx >= params.batch_size) {
        return;
    }

    uint K = params.kernel_size;
    uint state_len = params.state_length;
    uint L = params.input_length;

    // Base pointers for this specific thread
    uint weight_base = channel_idx * K;
    uint state_base = (batch_idx * params.channels + channel_idx) * state_len;
    uint data_base = (batch_idx * params.channels + channel_idx) * L;

    // Load kernel weights into registers (assuming small K)
    T w[32];
    for (uint j = 0; j < K; j++) {
        w[j] = weight[weight_base + j];
    }

    T b_val = params.has_bias ? bias[channel_idx] : (T)0.0f;

    // Optimized Sliding Window
    // We maintain the last K-1 elements in registers.
    T window[32];
    
    // Initialize window from conv_state or zeros
    if (params.has_conv_state) {
        for (uint j = 0; j < state_len; j++) {
            window[j] = conv_state[state_base + j];
        }
    } else {
        for (uint j = 0; j < state_len; j++) {
            window[j] = (T)0.0f;
        }
    }

    // Main loop over the sequence L
    for (uint i = 0; i < L; i++) {
        T current_input = input[data_base + i];
        
        // window contains elements [i-state_len, ..., i-1]
        // virtual_input consists of window + current_input
        // convolution is sum(virtual_input[j] * w[j])
        
        float acc = 0.0f;
        
        // Part A: Window contribution
        for (uint j = 0; j < state_len; j++) {
            acc += (float)window[j] * (float)w[j];
        }
        
        // Part B: Current input contribution
        acc += (float)current_input * (float)w[state_len];
        
        acc += (float)b_val;

        if (params.use_silu) {
            acc = (float)silu((float)acc);
        }

        output[data_base + i] = (T)acc;

        // Shift window: [w1, w2, w3] -> [w2, w3, current_input] for K=4
        for (uint j = 0; j < state_len - 1; j++) {
            window[j] = window[j+1];
        }
        if (state_len > 0) {
            window[state_len - 1] = current_input;
        }
    }

    // Write back present_state
    for (uint j = 0; j < state_len; j++) {
        present_state[state_base + j] = window[j];
    }
}

// Explicit instantiations for the bridge
kernel void causal_conv_with_state_fwd_float(
    device const float* input         [[buffer(0)]],
    device const float* weight        [[buffer(1)]],
    device const float* bias          [[buffer(2)]],
    device const float* conv_state    [[buffer(3)]],
    device float* output              [[buffer(4)]],
    device float* present_state       [[buffer(5)]],
    constant CausalConvParams& params [[buffer(6)]],
    uint2 tid                         [[thread_position_in_grid]])
{
    causal_conv_with_state_kernel<float>(input, weight, bias, conv_state, output, present_state, params, tid);
}

kernel void causal_conv_with_state_fwd_half(
    device const half* input          [[buffer(0)]],
    device const half* weight         [[buffer(1)]],
    device const half* bias           [[buffer(2)]],
    device const half* conv_state     [[buffer(3)]],
    device half* output               [[buffer(4)]],
    device half* present_state        [[buffer(5)]],
    constant CausalConvParams& params [[buffer(6)]],
    uint2 tid                         [[thread_position_in_grid]])
{
    causal_conv_with_state_kernel<half>(input, weight, bias, conv_state, output, present_state, params, tid);
}