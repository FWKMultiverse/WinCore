// Fused bias-add + GELU activation, forward and backward.
//
// Why this exists / why it can genuinely be faster than PyTorch eager:
// A normal PyTorch line like `torch.nn.functional.gelu(x + bias)` runs as
// TWO separate CUDA kernel launches: one for the add, one for GELU. Each
// kernel reads its input from VRAM and writes its full output back to
// VRAM before the next kernel starts. For a large activation tensor
// that round-trip to VRAM (not the compute itself) is the bottleneck --
// elementwise ops like this are memory-bandwidth-bound, not
// compute-bound.
//
// Fusing add+GELU into ONE kernel means the intermediate (x + bias)
// never touches VRAM -- it stays in a register for the few cycles
// between the add and the GELU. That is a genuine, measurable, well
// documented speedup for this exact pattern (this is the same fusion
// NVIDIA's own Megatron-LM and apex.fused_bias_gelu use). It is NOT a
// claim that this beats cuBLAS/cuDNN's matmul or convolution kernels --
// those are separate, compute-bound ops where NVIDIA's own hand-tuned
// kernels remain the fastest option, and nothing here disputes that.
//
// Dtype support
// -------------
// float32 / float64 / float16 / bfloat16 are templated below and run
// as GENUINE single-kernel-launch fused ops in that native dtype --
// same fusion, same speedup rationale as the original float32-only
// version, just parameterized over `scalar_t`. Math inside the kernel
// (tanh, polynomial terms) always accumulates in float32 for accuracy,
// then casts back to the tensor's native dtype on write -- the same
// approach PyTorch's own fp16 kernels use, so half-precision inputs
// don't lose extra accuracy to this kernel specifically.
//
// float8 (e4m3/e5m2) is handled differently and is NOT templated into
// this kernel: raw elementwise arithmetic on fp8 storage types isn't a
// standardized, portable CUDA operation the way it is for fp16/bf16 --
// real fp8 training paths (e.g. NVIDIA Transformer Engine) use
// specialized scaling/accumulation logic this reference kernel doesn't
// implement. The Python wrapper (fused_bias_gelu.py) instead upcasts
// fp8 input to float32, runs the float32 path below, and casts the
// result back to fp8 -- correct output, but WITHOUT the fusion speedup
// for that specific dtype (two extra cast kernels get launched). This
// is a documented, correctness-first bridge, not a claim of native fp8
// kernel support.
//
// This file has NOT been compiled or benchmarked in this sandbox --
// there is no GPU here. It needs a Windows machine with the CUDA
// Toolkit + MSVC Build Tools installed to build (see build.py in this
// folder) and an actual GPU to benchmark against the unfused version --
// and since this revision adds new dtype paths (fp64/fp16/bf16) that
// the original float32-only version never exercised, treat ALL of them
// as unverified until run for real, not just the float32 path that was
// tested before.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// tanh-approximation GELU (matches PyTorch's `approximate="tanh"` mode).
// Always computed in float32 for accuracy regardless of the tensor's
// storage dtype -- see file header.
__device__ __forceinline__ float gelu_fwd_f32(float x) {
    const float k0 = 0.7978845608028654f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * x3)));
}

// Derivative of the same tanh-approximation, needed for backward.
__device__ __forceinline__ float gelu_bwd_f32(float x) {
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float x2 = x * x;
    float inner = k0 * (x + k1 * x * x2);
    float t = tanhf(inner);
    float sech2 = 1.0f - t * t;
    float dinner_dx = k0 * (1.0f + 3.0f * k1 * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * dinner_dx;
}

template <typename scalar_t>
__global__ void fused_bias_gelu_fwd_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t col = idx % cols;
    // Widen to float32 for the add + GELU math, narrow back on write --
    // keeps fp16/bf16 accuracy on par with PyTorch's own fp16 GELU
    // rather than accumulating error in half precision.
    float v = static_cast<float>(x[idx]) + static_cast<float>(bias[col]);
    out[idx] = static_cast<scalar_t>(gelu_fwd_f32(v));
}

template <typename scalar_t>
__global__ void fused_bias_gelu_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ grad_x,
    int64_t rows,
    int64_t cols
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx >= total) return;
    int64_t col = idx % cols;
    float v = static_cast<float>(x[idx]) + static_cast<float>(bias[col]);
    float go = static_cast<float>(grad_out[idx]);
    grad_x[idx] = static_cast<scalar_t>(go * gelu_bwd_f32(v));
}

torch::Tensor fused_bias_gelu_fwd(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.size(-1) == bias.size(0), "bias size must match last dim of x");
    TORCH_CHECK(bias.scalar_type() == x.scalar_type(),
        "bias dtype must match x dtype (cast bias yourself if you need mixed dtypes)");

    auto x_c = x.contiguous();
    auto bias_c = bias.contiguous();
    auto out = torch::empty_like(x_c);
    int64_t cols = x_c.size(-1);
    int64_t rows = x_c.numel() / cols;

    const int threads = 256;
    const int64_t blocks = (rows * cols + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
        "fused_bias_gelu_fwd", ([&] {
            fused_bias_gelu_fwd_kernel<scalar_t><<<blocks, threads>>>(
                x_c.data_ptr<scalar_t>(), bias_c.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(), rows, cols);
        }));

    return out;
}

torch::Tensor fused_bias_gelu_bwd(torch::Tensor grad_out, torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(grad_out.scalar_type() == x.scalar_type(),
        "grad_out dtype must match x dtype");

    auto x_c = x.contiguous();
    auto bias_c = bias.contiguous();
    auto grad_out_c = grad_out.contiguous();
    auto grad_x = torch::empty_like(x_c);
    int64_t cols = x_c.size(-1);
    int64_t rows = x_c.numel() / cols;

    const int threads = 256;
    const int64_t blocks = (rows * cols + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x_c.scalar_type(),
        "fused_bias_gelu_bwd", ([&] {
            fused_bias_gelu_bwd_kernel<scalar_t><<<blocks, threads>>>(
                grad_out_c.data_ptr<scalar_t>(), x_c.data_ptr<scalar_t>(),
                bias_c.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(), rows, cols);
        }));

    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fused_bias_gelu_fwd, "Fused bias-add + GELU forward (CUDA)");
    m.def("bwd", &fused_bias_gelu_bwd, "Fused bias-add + GELU backward (CUDA)");
}
