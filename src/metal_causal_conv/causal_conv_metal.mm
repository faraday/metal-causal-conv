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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// ---------------------------------------------------------------------------
// Cached pipeline state
// ---------------------------------------------------------------------------

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pipeline = nil;
    bool initialized = false;
};

static MetalState& getState() {
    static MetalState state;
    return state;
}

static void ensureInitialized(const std::string& shader_path) {
    MetalState& state = getState();
    if (state.initialized) return;

    @autoreleasepool {
        state.device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(state.device != nil, "Metal is not available on this device");

        NSString* path = [NSString stringWithUTF8String:shader_path.c_str()];
        NSError* error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                    encoding:NSUTF8StringEncoding
                                                       error:&error];
        TORCH_CHECK(error == nil, "Failed to load Metal shader from: ", shader_path,
                    " — ", [[error localizedDescription] UTF8String]);

        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;
        state.library = [state.device newLibraryWithSource:source
                                                  options:options
                                                    error:&error];
        TORCH_CHECK(error == nil, "Failed to compile Metal shader: ",
                    [[error localizedDescription] UTF8String]);

        id<MTLFunction> function = [state.library newFunctionWithName:@"causal_conv_with_state_fwd"];
        TORCH_CHECK(function != nil,
                    "Metal function 'causal_conv_with_state_fwd' not found in shader");

        state.pipeline = [state.device newComputePipelineStateWithFunction:function
                                                                    error:&error];
        TORCH_CHECK(error == nil, "Failed to create compute pipeline: ",
                    [[error localizedDescription] UTF8String]);

        state.initialized = true;
    }
}

// ---------------------------------------------------------------------------
// Main dispatch function
// ---------------------------------------------------------------------------

std::vector<torch::Tensor> causal_conv_with_state_fwd(
    torch::Tensor input,        // (B, D, L)
    torch::Tensor weight,       // (D, 1, K)
    torch::Tensor bias,         // (D,) or empty
    torch::Tensor conv_state,   // (B, D, K-1) or empty
    bool has_bias,
    bool has_conv_state,
    bool use_silu,
    const std::string& shader_path
) {
    @autoreleasepool {
        ensureInitialized(shader_path);
        MetalState& state = getState();

        // Validate MPS device and contiguity
        TORCH_CHECK(input.is_mps(), "input must be on MPS device");
        TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

        uint32_t B = input.size(0);
        uint32_t D = input.size(1);
        uint32_t L = input.size(2);
        uint32_t K = weight.size(2);
        uint32_t state_len = K - 1;
        uint32_t total_output = B * D * L;

        // Allocate outputs on MPS
        auto output = torch::empty({B, D, L}, input.options());
        auto present_state = torch::empty({B, D, (int64_t)state_len}, input.options());

        // Dummy buffer for optional inputs that are not provided
        auto dummy = torch::zeros({1}, input.options());

        uint32_t has_bias_val = has_bias ? 1 : 0;
        uint32_t has_state_val = has_conv_state ? 1 : 0;
        uint32_t use_silu_val = use_silu ? 1 : 0;

        // Get the active PyTorch MPS compute encoder
        id<MTLComputeCommandEncoder> encoder = at::mps::getCurrentMPSStream()->commandEncoder();
        TORCH_CHECK(encoder != nil, "Failed to get PyTorch active compute encoder");

        [encoder setComputePipelineState:state.pipeline];

        // Bind input buffers
        [encoder setBuffer:getMTLBufferStorage(input)
                    offset:input.storage_offset() * sizeof(float) atIndex:0];
        [encoder setBuffer:getMTLBufferStorage(weight)
                    offset:weight.storage_offset() * sizeof(float) atIndex:1];

        // Optional: bias
        [encoder setBuffer:has_bias ? getMTLBufferStorage(bias) : getMTLBufferStorage(dummy)
                    offset:has_bias ? bias.storage_offset() * sizeof(float) : 0
                   atIndex:2];

        // Optional: conv_state
        [encoder setBuffer:has_conv_state ? getMTLBufferStorage(conv_state) : getMTLBufferStorage(dummy)
                    offset:has_conv_state ? conv_state.storage_offset() * sizeof(float) : 0
                   atIndex:3];

        // Output buffers
        [encoder setBuffer:getMTLBufferStorage(output)
                    offset:output.storage_offset() * sizeof(float) atIndex:4];
        [encoder setBuffer:getMTLBufferStorage(present_state)
                    offset:present_state.storage_offset() * sizeof(float) atIndex:5];

        // Scalar uniforms via setBytes
        [encoder setBytes:&B         length:sizeof(uint32_t) atIndex:6];
        [encoder setBytes:&D         length:sizeof(uint32_t) atIndex:7];
        [encoder setBytes:&L         length:sizeof(uint32_t) atIndex:8];
        [encoder setBytes:&K         length:sizeof(uint32_t) atIndex:9];
        [encoder setBytes:&state_len length:sizeof(uint32_t) atIndex:10];
        [encoder setBytes:&total_output length:sizeof(uint32_t) atIndex:11];
        [encoder setBytes:&has_bias_val  length:sizeof(uint32_t) atIndex:12];
        [encoder setBytes:&has_state_val length:sizeof(uint32_t) atIndex:13];
        [encoder setBytes:&use_silu_val  length:sizeof(uint32_t) atIndex:14];

        // Dispatch: one thread per (batch, channel, position) triple
        NSUInteger threadGroupSize = MIN(state.pipeline.maxTotalThreadsPerThreadgroup,
                                         (NSUInteger)total_output);
        if (threadGroupSize > 256) threadGroupSize = 256;

        MTLSize gridSize = MTLSizeMake(total_output, 1, 1);
        MTLSize groupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:groupSize];

        // Note: we DO NOT call [encoder endEncoding] or [commandBuffer commit].
        // PyTorch manages this encoder and will end/commit it automatically.

        return {output, present_state};
    }
}

// ---------------------------------------------------------------------------
// Pybind11 module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv_with_state_fwd", &causal_conv_with_state_fwd,
          "Fused CausalConvWithState forward pass on Metal GPU",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("conv_state"),
          py::arg("has_bias"),
          py::arg("has_conv_state"),
          py::arg("use_silu"),
          py::arg("shader_path"));
}
