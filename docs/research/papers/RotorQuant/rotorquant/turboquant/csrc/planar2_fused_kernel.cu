/*
 * PlanarQuant: Fused 2D planar rotation + Lloyd-Max quantization kernel.
 *
 * For each 2D pair in the embedding:
 *   1. Apply SO(2) rotation: (cos θ, sin θ)
 *   2. Quantize each scalar to nearest Lloyd-Max centroid
 *   3. Apply inverse rotation (transpose)
 *
 * This is the 2D analogue of IsoQuant's 4D quaternion kernel.
 * Only 4 FMAs per pair (vs 16 for quaternion, vs 28 for Clifford).
 *
 * Reference: ParaMind2025/isoquant/csrc/planar2_fused_kernel.cu
 */

#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_LEVELS 256

/* ── Type conversion helpers ─────────────────────────────────────── */

template <typename T>
__device__ __forceinline__ float convert_to_float(T value) { return 0.0f; }
template <> __device__ __forceinline__ float convert_to_float<c10::Half>(c10::Half value) { return __half2float(value); }
template <> __device__ __forceinline__ float convert_to_float<float>(float value) { return value; }
template <> __device__ __forceinline__ float convert_to_float<at::BFloat16>(at::BFloat16 value) { return static_cast<float>(value); }

template <typename T>
__device__ __forceinline__ T convert_from_float(float value) { return static_cast<T>(0); }
template <> __device__ __forceinline__ c10::Half convert_from_float<c10::Half>(float value) { return __float2half(value); }
template <> __device__ __forceinline__ float convert_from_float<float>(float value) { return value; }
template <> __device__ __forceinline__ at::BFloat16 convert_from_float<at::BFloat16>(float value) { return static_cast<at::BFloat16>(value); }

/* ── 2D rotation primitives ──────────────────────────────────────── */

__device__ __forceinline__ void rot2_apply(float c, float s, const float v[2], float out[2]) {
    out[0] = c * v[0] - s * v[1];
    out[1] = s * v[0] + c * v[1];
}

__device__ __forceinline__ void rot2_inverse(float c, float s, const float v[2], float out[2]) {
    out[0] = c * v[0] + s * v[1];
    out[1] = -s * v[0] + c * v[1];
}

/* ── Scalar quantization (nearest centroid) ──────────────────────── */

__device__ __forceinline__ float quantize_scalar(float val, const float* __restrict__ centroids, int levels) {
    float best = centroids[0];
    float min_d = fabsf(val - best);
    for (int i = 1; i < levels; ++i) {
        float d = fabsf(val - centroids[i]);
        if (d < min_d) {
            min_d = d;
            best = centroids[i];
        }
    }
    return best;
}

template <int LEVELS>
__device__ __forceinline__ float quantize_scalar_fixed(float val, const float* __restrict__ centroids) {
    float best = centroids[0];
    float min_d = fabsf(val - best);
    #pragma unroll
    for (int i = 1; i < LEVELS; ++i) {
        float cand = centroids[i];
        float d = fabsf(val - cand);
        if (d < min_d) {
            min_d = d;
            best = cand;
        }
    }
    return best;
}

/* ── Main fused kernel ───────────────────────────────────────────── */

template <typename T, int LEVELS>
__global__ void planar2_fused_kernel(
    const T* __restrict__ input,
    const float* __restrict__ rot2,       // (n_groups, 2) as [cos θ, sin θ]
    const float* __restrict__ centroids,  // (n_levels,)
    T* __restrict__ output,
    int batch_size,
    int emb_dim,
    int n_groups,
    int n_levels)
{
    // Load centroids into shared memory for fast repeated access
    __shared__ float sh_centroids[MAX_LEVELS];

    int tid = threadIdx.x;
    for (int i = tid; i < n_levels; i += blockDim.x) {
        sh_centroids[i] = centroids[i];
    }
    __syncthreads();

    int b = blockIdx.x;
    if (b >= batch_size) return;

    const T* in_ptr = input + b * emb_dim;
    T* out_ptr = output + b * emb_dim;

    // Each thread handles one 2D group
    for (int g = tid; g < n_groups; g += blockDim.x) {
        int base = g * 2;

        // Load pair (with bounds check for odd dimensions)
        float v[2] = {0.f, 0.f};
        if (base < emb_dim) v[0] = convert_to_float(in_ptr[base]);
        if (base + 1 < emb_dim) v[1] = convert_to_float(in_ptr[base + 1]);

        // Load rotation for this group
        float c = rot2[g * 2 + 0];
        float s = rot2[g * 2 + 1];

        // Forward rotation
        float rotated[2];
        rot2_apply(c, s, v, rotated);

        // Quantize each scalar
        float qv[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            if constexpr (LEVELS > 0) {
                qv[i] = quantize_scalar_fixed<LEVELS>(rotated[i], sh_centroids);
            } else {
                qv[i] = quantize_scalar(rotated[i], sh_centroids, n_levels);
            }
        }

        // Inverse rotation
        float restored[2];
        rot2_inverse(c, s, qv, restored);

        // Store result
        if (base < emb_dim) out_ptr[base] = convert_from_float<T>(restored[0]);
        if (base + 1 < emb_dim) out_ptr[base + 1] = convert_from_float<T>(restored[1]);
    }
}

/* ── Dispatch helpers ────────────────────────────────────────────── */

template <typename T, int LEVELS>
torch::Tensor planar2_fused_impl(
    torch::Tensor input,
    torch::Tensor rot2,
    torch::Tensor centroids,
    int n_levels)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 1) / 2;

    auto output = torch::empty_like(input);
    int threads = min(256, max(n_groups, WARP_SIZE));

    planar2_fused_kernel<T, LEVELS><<<batch_size, threads>>>(
        input.data_ptr<T>(),
        rot2.data_ptr<float>(),
        centroids.data_ptr<float>(),
        output.data_ptr<T>(),
        batch_size,
        emb_dim,
        n_groups,
        n_levels);
    return output;
}

template <typename T>
torch::Tensor planar2_fused_dispatch(
    torch::Tensor input,
    torch::Tensor rot2,
    torch::Tensor centroids,
    int n_levels)
{
    // Compile-time unroll for common bit-widths (4=2bit, 8=3bit, 16=4bit)
    switch (n_levels) {
        case 4:
            return planar2_fused_impl<T, 4>(input, rot2, centroids, n_levels);
        case 8:
            return planar2_fused_impl<T, 8>(input, rot2, centroids, n_levels);
        case 16:
            return planar2_fused_impl<T, 16>(input, rot2, centroids, n_levels);
        default:
            return planar2_fused_impl<T, 0>(input, rot2, centroids, n_levels);
    }
}

/* ── Python bindings ─────────────────────────────────────────────── */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("planar2_fused_float", &planar2_fused_dispatch<float>,
          "Fused planar 2D rotation + quantization (float32)");
    m.def("planar2_fused_half", &planar2_fused_dispatch<c10::Half>,
          "Fused planar 2D rotation + quantization (float16)");
    m.def("planar2_fused_bf16", &planar2_fused_dispatch<at::BFloat16>,
          "Fused planar 2D rotation + quantization (bfloat16)");
}
