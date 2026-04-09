/******************************************************************************
 * Objective-C++ bridge for the Metal CausalConvWithState kernel.
 *
 * Loads the .metal shader, creates a compute pipeline, extracts MTLBuffers
 * from PyTorch MPS tensors, encodes the dispatch, and returns.
 *
 * SYNCHRONIZATION: Integrates natively with PyTorch's MPSStream.
 * By using PyTorch's own computeCommandEncoder, we achieve zero-overhead
 * asynchronous dispatch — no manual synchronization needed.
 *
 * Build: compiled as part of a PyTorch CppExtension via setup.py
 ******************************************************************************/

#include <ATen/mps/MPSStream.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Struct Definitions (Must align with Metal shader)
// ---------------------------------------------------------------------------

struct alignas(4) CausalConvParams {
  uint32_t batch_size;
  uint32_t channels;
  uint32_t input_length;
  uint32_t kernel_size;
  uint32_t state_length;
  uint32_t has_bias;
  uint32_t has_conv_state;
  uint32_t use_silu;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor &tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------

struct MetalState {
  id<MTLDevice> device = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> pipeline_float = nil;
  id<MTLComputePipelineState> pipeline_half = nil;
  bool initialized = false;
};

static MetalState &getState() {
  static MetalState state;
  return state;
}

static void ensureInitialized(const std::string &shader_path) {
  MetalState &state = getState();
  if (state.initialized)
    return;

  @autoreleasepool {
    state.device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(state.device != nil, "Metal is not available on this device");

    NSString *path = [NSString stringWithUTF8String:shader_path.c_str()];
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:path
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    TORCH_CHECK(error == nil, "Failed to load Metal shader from: ", shader_path,
                " — ", [[error localizedDescription] UTF8String]);

    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    state.library = [state.device newLibraryWithSource:source
                                               options:options
                                                 error:&error];
    TORCH_CHECK(error == nil, "Failed to compile Metal shader: ",
                [[error localizedDescription] UTF8String]);

    id<MTLFunction> func_float =
        [state.library newFunctionWithName:@"causal_conv_with_state_fwd_float"];
    id<MTLFunction> func_half =
        [state.library newFunctionWithName:@"causal_conv_with_state_fwd_half"];
    
    TORCH_CHECK(func_float != nil && func_half != nil,
                "Metal functions not found in shader");

    state.pipeline_float = [state.device newComputePipelineStateWithFunction:func_float
                                                                       error:&error];
    TORCH_CHECK(error == nil, "Failed to create float pipeline");
    
    state.pipeline_half = [state.device newComputePipelineStateWithFunction:func_half
                                                                      error:&error];
    TORCH_CHECK(error == nil, "Failed to create half pipeline");

    state.initialized = true;
  }
}

// ---------------------------------------------------------------------------
// Main dispatch function
// ---------------------------------------------------------------------------

std::vector<torch::Tensor>
causal_conv_with_state_fwd(torch::Tensor input,      // (B, D, L)
                           torch::Tensor weight,     // (D, 1, K)
                           torch::Tensor bias,       // (D,) or empty
                           torch::Tensor conv_state, // (B, D, K-1) or empty
                           bool has_bias, bool has_conv_state, bool use_silu,
                           const std::string &shader_path) {
  @autoreleasepool {
    ensureInitialized(shader_path);
    MetalState &state = getState();

    // Validate MPS device and contiguity
    TORCH_CHECK(input.is_mps(), "input must be on MPS device");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    uint32_t B = input.size(0);
    uint32_t D = input.size(1);
    uint32_t L = input.size(2);
    uint32_t K = weight.size(2);
    uint32_t state_len = K - 1;

    // Allocate outputs on MPS
    auto output = torch::empty({B, D, L}, input.options());
    auto present_state =
        torch::empty({B, D, (int64_t)state_len}, input.options());

    // Dummy buffer to safely bind to the encoder if optionals are missing
    auto dummy = torch::zeros({1}, input.options());

    // Pack parameters into struct
    CausalConvParams params;
    params.batch_size = B;
    params.channels = D;
    params.input_length = L;
    params.kernel_size = K;
    params.state_length = state_len;
    params.has_bias = has_bias ? 1 : 0;
    params.has_conv_state = has_conv_state ? 1 : 0;
    params.use_silu = use_silu ? 1 : 0;

    // Select pipeline based on dtype
    id<MTLComputePipelineState> pso = (input.scalar_type() == torch::kHalf) 
                                       ? state.pipeline_half 
                                       : state.pipeline_float;

    // Get the active PyTorch MPS compute encoder
    id<MTLComputeCommandEncoder> encoder =
        at::mps::getCurrentMPSStream()->commandEncoder();
    TORCH_CHECK(encoder != nil, "Failed to get PyTorch active compute encoder");

    [encoder setComputePipelineState:pso];

    size_t element_size = (input.scalar_type() == torch::kHalf) ? 2 : 4;

    // Bind input and output buffers
    [encoder setBuffer:getMTLBufferStorage(input)
                offset:input.storage_offset() * element_size
               atIndex:0];
    [encoder setBuffer:getMTLBufferStorage(weight)
                offset:weight.storage_offset() * element_size
               atIndex:1];

    [encoder setBuffer:has_bias ? getMTLBufferStorage(bias)
                                : getMTLBufferStorage(dummy)
                offset:has_bias ? bias.storage_offset() * element_size : 0
               atIndex:2];

    [encoder
        setBuffer:has_conv_state ? getMTLBufferStorage(conv_state)
                                 : getMTLBufferStorage(dummy)
           offset:has_conv_state ? conv_state.storage_offset() * element_size
                                 : 0
          atIndex:3];

    [encoder setBuffer:getMTLBufferStorage(output)
                offset:output.storage_offset() * element_size
               atIndex:4];
    [encoder setBuffer:getMTLBufferStorage(present_state)
                offset:present_state.storage_offset() * element_size
               atIndex:5];

    // Set unified struct via a single setBytes call
    [encoder setBytes:&params length:sizeof(CausalConvParams) atIndex:6];

    // Setup 3D Grid: (Length, Channels, Batch)
    MTLSize gridSize = MTLSizeMake(L, D, B);

    // Optimize Threadgroup Size
    // We want a large N in the L dimension to maximize shared memory reuse.
    // However, N must not exceed MAX_TILE_SIZE (512) or hardware limits.
    NSUInteger maxThreads = pso.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_x = MIN((NSUInteger)L, MIN((NSUInteger)256, maxThreads)); 
    NSUInteger tg_y = 1; // 1 channel per group for simplicity and shared memory alignment
    NSUInteger tg_z = 1;
    MTLSize groupSize = MTLSizeMake(tg_x, tg_y, tg_z);

    // Dispatch
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

    return {output, present_state};
  }
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("causal_conv_with_state_fwd", &causal_conv_with_state_fwd,
        "Fused CausalConvWithState forward pass on Metal GPU", py::arg("input"),
        py::arg("weight"), py::arg("bias"), py::arg("conv_state"),
        py::arg("has_bias"), py::arg("has_conv_state"), py::arg("use_silu"),
        py::arg("shader_path"));
}