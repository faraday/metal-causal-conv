/******************************************************************************
 * Metal CausalConvWithState — Tiled Shared-Memory Fused Optimization
 *
 * Parallelizes across BOTH Sequence Length (L) and Channels (D).
 * Uses Threadgroup Memory (Shared Memory) to cache input motifs, avoiding
 * redundant global memory reads and maximizing GPU occupancy.
 *
 * Each thread computes ONE output element.
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

// We use a fixed maximum tile size for shared memory to simplify allocation.
// For L=256, a tile of 256 + 3 is plenty.
#define MAX_TILE_SIZE 512

template<typename T>
void causal_conv_tiled_impl(
    device const T* input,
    device const T* weight,
    device const T* bias,
    device const T* conv_state,
    device T* output,
    device T* present_state,
    constant CausalConvParams& params,
    threadgroup T* s_input,
    uint3 gid,
    uint3 tid,
    uint3 tgid,
    uint3 threads_per_group)
{
    // Hierarchical indices
    const uint pos = gid.x;
    const uint channel_idx = gid.y;
    const uint batch_idx = gid.z;

    const uint l_tid = tid.x; // Local thread ID in the tile
    const uint K = params.kernel_size;
    const uint state_len = params.state_length;
    const uint L = params.input_length;

    // Base pointers
    const uint weight_base = channel_idx * K;
    const uint state_base = (batch_idx * params.channels + channel_idx) * state_len;
    const uint data_base = (batch_idx * params.channels + channel_idx) * L;

    // 1. Cooperative Loading of Input + Halo into Shared Memory
    // Each threadgroup handles a block of N threads in the 'L' dimension.
    // We need to load input[start...end] AND input[start-state_len...start-1]
    
    const uint tile_start = tgid.x * threads_per_group.x;
    
    // Load main data
    if (pos < L) {
        s_input[state_len + l_tid] = input[data_base + pos];
    }
    
    // Load halo (the state_len elements before this tile)
    if (l_tid < state_len) {
        if (tile_start == 0) {
            // First tile: load from conv_state
            if (params.has_conv_state) {
                s_input[l_tid] = conv_state[state_base + l_tid];
            } else {
                s_input[l_tid] = (T)0.0f;
            }
        } else {
            // Subsequent tile: load from previous global input data
            s_input[l_tid] = input[data_base + tile_start - state_len + l_tid];
        }
    }

    // Synchronize to ensure all data is in shared memory
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (pos >= L || channel_idx >= params.channels || batch_idx >= params.batch_size) {
        return;
    }

    // 2. Convolution Computation
    // All inputs are now in s_input. Weights are small enough for registers.
    T w[32];
    for (uint j = 0; j < K; j++) {
        w[j] = weight[weight_base + j];
    }

    float acc = 0.0f;
    for (uint j = 0; j < K; j++) {
        acc += (float)s_input[l_tid + j] * (float)w[j];
    }

    if (params.has_bias) {
        acc += (float)bias[channel_idx];
    }

    if (params.use_silu) {
        acc = (float)silu((float)acc);
    }

    output[data_base + pos] = (T)acc;

    // 3. State Update
    // The very last threads of the sequence write back the present_state.
    if (pos == L - 1) {
        for (uint j = 0; j < state_len; j++) {
            // present_state is the last (K-1) elements of the virtual input
            // virtual input at the end is s_input[l_tid + 1 ... l_tid + state_len]
            present_state[state_base + j] = s_input[l_tid + 1 + j];
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
    uint3 gid                         [[thread_position_in_grid]],
    uint3 tid                         [[thread_position_in_threadgroup]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint3 threads_per_group           [[threads_per_threadgroup]])
{
    threadgroup float s_input[MAX_TILE_SIZE + 32];
    causal_conv_tiled_impl<float>(input, weight, bias, conv_state, output, present_state, params, s_input, gid, tid, tgid, threads_per_group);
}

kernel void causal_conv_with_state_fwd_half(
    device const half* input          [[buffer(0)]],
    device const half* weight         [[buffer(1)]],
    device const half* bias           [[buffer(2)]],
    device const half* conv_state     [[buffer(3)]],
    device half* output               [[buffer(4)]],
    device half* present_state        [[buffer(5)]],
    constant CausalConvParams& params [[buffer(6)]],
    uint3 gid                         [[thread_position_in_grid]],
    uint3 tid                         [[thread_position_in_threadgroup]],
    uint3 tgid                        [[threadgroup_position_in_grid]],
    uint3 threads_per_group           [[threads_per_threadgroup]])
{
    threadgroup half s_input[MAX_TILE_SIZE + 32];
    causal_conv_tiled_impl<half>(input, weight, bias, conv_state, output, present_state, params, s_input, gid, tid, tgid, threads_per_group);
}