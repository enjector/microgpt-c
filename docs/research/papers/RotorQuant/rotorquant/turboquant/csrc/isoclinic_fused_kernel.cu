/**
 * IsoQuant: Fused CUDA kernels for quaternion 4D block rotation + quantization.
 *
 * Two variants:
 *   isoclinic_full_fused — T(v) = q_L v conj(q_R) → quantize → conj(q_L) v̂ q_R
 *   isoclinic_fast_fused — T(v) = q_L v → quantize → conj(q_L) v̂
 *
 * Template specialization for LEVELS (4, 8, 16, or dynamic) enables
 * compile-time unrolled quantization loops for common bit widths.
 *
 * Reference: Ji, "IsoQuant: Hardware-Aligned SO(4) Isoclinic Rotations
 *            for LLM KV Cache Compression", March 2026.
 */

#include <torch/extension.h>
#include <cmath>

#define WARP_SIZE 32
#define MAX_LEVELS 256

template <typename T>
__device__ float convert_to_float(T value) { return 0.0f; }
template <> __device__ float convert_to_float<c10::Half>(c10::Half value) { return __half2float(value); }
template <> __device__ float convert_to_float<float>(float value) { return value; }
template <> __device__ float convert_to_float<at::BFloat16>(at::BFloat16 value) { return static_cast<float>(value); }

template <typename T>
__device__ T convert_from_float(float value) { return static_cast<T>(0); }
template <> __device__ c10::Half convert_from_float<c10::Half>(float value) { return __float2half(value); }
template <> __device__ float convert_from_float<float>(float value) { return value; }
template <> __device__ at::BFloat16 convert_from_float<at::BFloat16>(float value) { return static_cast<at::BFloat16>(value); }

__device__ inline void quat_conj(const float q[4], float out[4]) {
    out[0] = q[0];
    out[1] = -q[1];
    out[2] = -q[2];
    out[3] = -q[3];
}

__device__ inline void quat_mul(const float a[4], const float b[4], float out[4]) {
    out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

__device__ inline float quantize_scalar(float val, const float* __restrict__ centroids, int levels) {
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

// ── IsoQuant-Full: q_L v conj(q_R) → quantize → conj(q_L) v̂ q_R ────

template <typename T, int LEVELS>
__global__ void isoclinic_full_fused_kernel(
    const T* __restrict__ input,
    const float* __restrict__ q_left,
    const float* __restrict__ q_right,
    const float* __restrict__ centroids,
    T* __restrict__ output,
    int batch_size,
    int emb_dim,
    int n_groups,
    int n_levels)
{
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

    for (int g = tid; g < n_groups; g += blockDim.x) {
        int base = g * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (base < emb_dim) v[0] = convert_to_float(in_ptr[base]);
        if (base + 1 < emb_dim) v[1] = convert_to_float(in_ptr[base + 1]);
        if (base + 2 < emb_dim) v[2] = convert_to_float(in_ptr[base + 2]);
        if (base + 3 < emb_dim) v[3] = convert_to_float(in_ptr[base + 3]);

        float ql[4], qr[4], qr_conj[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ql[i] = q_left[g * 4 + i];
            qr[i] = q_right[g * 4 + i];
        }
        quat_conj(qr, qr_conj);

        float temp[4], rotated[4];
        quat_mul(ql, v, temp);
        quat_mul(temp, qr_conj, rotated);

        float qv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (LEVELS > 0) {
                qv[i] = quantize_scalar_fixed<LEVELS>(rotated[i], sh_centroids);
            } else {
                qv[i] = quantize_scalar(rotated[i], sh_centroids, n_levels);
            }
        }

        float ql_conj[4], temp2[4], restored[4];
        quat_conj(ql, ql_conj);
        quat_mul(ql_conj, qv, temp2);
        quat_mul(temp2, qr, restored);

        if (base < emb_dim) out_ptr[base] = convert_from_float<T>(restored[0]);
        if (base + 1 < emb_dim) out_ptr[base + 1] = convert_from_float<T>(restored[1]);
        if (base + 2 < emb_dim) out_ptr[base + 2] = convert_from_float<T>(restored[2]);
        if (base + 3 < emb_dim) out_ptr[base + 3] = convert_from_float<T>(restored[3]);
    }
}

// ── IsoQuant-Fast: q_L v → quantize → conj(q_L) v̂ ──────────────────

template <typename T, int LEVELS>
__global__ void isoclinic_fast_fused_kernel(
    const T* __restrict__ input,
    const float* __restrict__ q_left,
    const float* __restrict__ centroids,
    T* __restrict__ output,
    int batch_size,
    int emb_dim,
    int n_groups,
    int n_levels)
{
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

    for (int g = tid; g < n_groups; g += blockDim.x) {
        int base = g * 4;
        float v[4] = {0.f, 0.f, 0.f, 0.f};
        if (base < emb_dim) v[0] = convert_to_float(in_ptr[base]);
        if (base + 1 < emb_dim) v[1] = convert_to_float(in_ptr[base + 1]);
        if (base + 2 < emb_dim) v[2] = convert_to_float(in_ptr[base + 2]);
        if (base + 3 < emb_dim) v[3] = convert_to_float(in_ptr[base + 3]);

        float ql[4], ql_conj[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ql[i] = q_left[g * 4 + i];
        }
        quat_conj(ql, ql_conj);

        float rotated[4];
        quat_mul(ql, v, rotated);

        float qv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if constexpr (LEVELS > 0) {
                qv[i] = quantize_scalar_fixed<LEVELS>(rotated[i], sh_centroids);
            } else {
                qv[i] = quantize_scalar(rotated[i], sh_centroids, n_levels);
            }
        }

        float restored[4];
        quat_mul(ql_conj, qv, restored);

        if (base < emb_dim) out_ptr[base] = convert_from_float<T>(restored[0]);
        if (base + 1 < emb_dim) out_ptr[base + 1] = convert_from_float<T>(restored[1]);
        if (base + 2 < emb_dim) out_ptr[base + 2] = convert_from_float<T>(restored[2]);
        if (base + 3 < emb_dim) out_ptr[base + 3] = convert_from_float<T>(restored[3]);
    }
}

// ── Dispatch: template specialization for common bit widths ──────────

template <typename T, int LEVELS>
torch::Tensor isoclinic_full_fused_impl(
    torch::Tensor input, torch::Tensor q_left, torch::Tensor q_right,
    torch::Tensor centroids, int n_levels)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 3) / 4;
    auto output = torch::empty_like(input);
    int threads = min(256, max(n_groups, WARP_SIZE));

    isoclinic_full_fused_kernel<T, LEVELS><<<batch_size, threads>>>(
        input.data_ptr<T>(), q_left.data_ptr<float>(), q_right.data_ptr<float>(),
        centroids.data_ptr<float>(), output.data_ptr<T>(),
        batch_size, emb_dim, n_groups, n_levels);
    return output;
}

template <typename T, int LEVELS>
torch::Tensor isoclinic_fast_fused_impl(
    torch::Tensor input, torch::Tensor q_left,
    torch::Tensor centroids, int n_levels)
{
    int batch_size = input.size(0);
    int emb_dim = input.size(1);
    int n_groups = (emb_dim + 3) / 4;
    auto output = torch::empty_like(input);
    int threads = min(256, max(n_groups, WARP_SIZE));

    isoclinic_fast_fused_kernel<T, LEVELS><<<batch_size, threads>>>(
        input.data_ptr<T>(), q_left.data_ptr<float>(),
        centroids.data_ptr<float>(), output.data_ptr<T>(),
        batch_size, emb_dim, n_groups, n_levels);
    return output;
}

template <typename T>
torch::Tensor isoclinic_full_fused_dispatch(
    torch::Tensor input, torch::Tensor q_left, torch::Tensor q_right,
    torch::Tensor centroids, int n_levels)
{
    switch (n_levels) {
        case 4:  return isoclinic_full_fused_impl<T, 4>(input, q_left, q_right, centroids, n_levels);
        case 8:  return isoclinic_full_fused_impl<T, 8>(input, q_left, q_right, centroids, n_levels);
        case 16: return isoclinic_full_fused_impl<T, 16>(input, q_left, q_right, centroids, n_levels);
        default: return isoclinic_full_fused_impl<T, 0>(input, q_left, q_right, centroids, n_levels);
    }
}

template <typename T>
torch::Tensor isoclinic_fast_fused_dispatch(
    torch::Tensor input, torch::Tensor q_left,
    torch::Tensor centroids, int n_levels)
{
    switch (n_levels) {
        case 4:  return isoclinic_fast_fused_impl<T, 4>(input, q_left, centroids, n_levels);
        case 8:  return isoclinic_fast_fused_impl<T, 8>(input, q_left, centroids, n_levels);
        case 16: return isoclinic_fast_fused_impl<T, 16>(input, q_left, centroids, n_levels);
        default: return isoclinic_fast_fused_impl<T, 0>(input, q_left, centroids, n_levels);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("isoclinic_full_fused_float", &isoclinic_full_fused_dispatch<float>);
    m.def("isoclinic_full_fused_half", &isoclinic_full_fused_dispatch<c10::Half>);
    m.def("isoclinic_full_fused_bf16", &isoclinic_full_fused_dispatch<at::BFloat16>);

    m.def("isoclinic_fast_fused_float", &isoclinic_fast_fused_dispatch<float>);
    m.def("isoclinic_fast_fused_half", &isoclinic_fast_fused_dispatch<c10::Half>);
    m.def("isoclinic_fast_fused_bf16", &isoclinic_fast_fused_dispatch<at::BFloat16>);
}
