// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file spline_inner.cu
 *
 *  \brief CUDA kernels to perform operations on splines.
 */

#include "../SDDK/GPU/cuda_common.hpp"
#include "../SDDK/GPU/acc_runtime.hpp"

#if __CUDA_ARCH__ < 600
__device__ double atomicAddCustom(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

template <unsigned int BLOCK_SIZE>
__global__ void spline_inner_product_gpu_kernel_m0(const int* __restrict__ num_points, const int* __restrict__ offsets,
                                                   const double4* __restrict__ coeffs_1,
                                                   const double4* __restrict__ coeffs_2, const double* __restrict__ x0,
                                                   const double* __restrict__ dx, double* __restrict__ results)
{
    // static_assert(BLOCK_SIZE >= 64, "block size requirement for reduction not fulfilled!");
    assert(BLOCK_SIZE == blockDim.x);

    __shared__ double sdata[BLOCK_SIZE];
    const auto tid    = threadIdx.x;
    const auto i    = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    const auto offset = offsets[blockIdx.y];
    const auto N      = num_points[blockIdx.y];

    double sum = 0.0;
    if (i < N - 1) {
        const double4 f_coeff = coeffs_1[offset + i];
        const double4 g_coeff = coeffs_2[offset + i];
        const double dx_i     = dx[offset + i];

        const double faga = f_coeff.x * g_coeff.x;
        const double fdgd = f_coeff.w * g_coeff.w;

        const double k1 = f_coeff.x * g_coeff.y + f_coeff.y * g_coeff.x;
        const double k2 = f_coeff.z * g_coeff.x + f_coeff.y * g_coeff.y + f_coeff.x * g_coeff.z;
        const double k3 = f_coeff.x * g_coeff.w + f_coeff.y * g_coeff.z + f_coeff.z * g_coeff.y + f_coeff.w * g_coeff.x;
        const double k4 = f_coeff.y * g_coeff.w + f_coeff.z * g_coeff.z + f_coeff.w * g_coeff.y;
        const double k5 = f_coeff.z * g_coeff.w + f_coeff.w * g_coeff.z;

        sum += dx_i *
               (faga + dx_i * (k1 / 2.0 +
                               dx_i * (k2 / 3.0 +
                                       dx_i * (k3 / 4.0 + dx_i * (k4 / 5.0 + dx_i * (k5 / 6.0 + dx_i * fdgd / 7.0))))));
    }

    sdata[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        for (unsigned i = 1; i < BLOCK_SIZE; ++i) {
            sum += sdata[i];
        }
#if __CUDA_ARCH__ < 600
        atomicAddCustom(results + blockIdx.y, sum);
#else
        atomicAdd(results + blockIdx.y, sum);
#endif
    }
}

extern "C" void spline_inner_product_gpu_m0(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results)
{
    constexpr unsigned int block_size = 256;
    dim3 threadBlock(block_size);
    dim3 blockGrid((max_num_points + threadBlock.x - 1) / threadBlock.x, num_splines);

    acc::zero(results, num_splines, stream);
    accLaunchKernel(spline_inner_product_gpu_kernel_m0<block_size>, blockGrid, threadBlock, 0, stream, num_points, offsets,
                    coeffs_1, coeffs_2, x0, dx, results);
}

template <unsigned int BLOCK_SIZE>
__global__ void spline_inner_product_gpu_kernel_m1(const int* __restrict__ num_points, const int* __restrict__ offsets,
                                                   const double4* __restrict__ coeffs_1,
                                                   const double4* __restrict__ coeffs_2, const double* __restrict__ x0,
                                                   const double* __restrict__ dx, double* __restrict__ results)
{
    // static_assert(BLOCK_SIZE >= 64, "block size requirement for reduction not fulfilled!");
    assert(BLOCK_SIZE == blockDim.x);

    __shared__ double sdata[BLOCK_SIZE];
    const auto tid    = threadIdx.x;
    const auto i    = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    const auto offset = offsets[blockIdx.y];
    const auto N      = num_points[blockIdx.y];

    double sum = 0.0;
    if (i < N - 1) {
        const double4 f_coeff = coeffs_1[offset + i];
        const double4 g_coeff = coeffs_2[offset + i];
        const double dx_i     = dx[offset + i];
        const double x0_i     = x0[offset + i];

        const double faga = f_coeff.x * g_coeff.x;
        const double fdgd = f_coeff.w * g_coeff.w;

        const double k1 = f_coeff.x * g_coeff.y + f_coeff.y * g_coeff.x;
        const double k2 = f_coeff.z * g_coeff.x + f_coeff.y * g_coeff.y + f_coeff.x * g_coeff.z;
        const double k3 = f_coeff.x * g_coeff.w + f_coeff.y * g_coeff.z + f_coeff.z * g_coeff.y + f_coeff.w * g_coeff.x;
        const double k4 = f_coeff.y * g_coeff.w + f_coeff.z * g_coeff.z + f_coeff.w * g_coeff.y;
        const double k5 = f_coeff.z * g_coeff.w + f_coeff.w * g_coeff.z;

        sum +=
            dx_i *
            ((faga * x0_i) +
             dx_i * ((faga + k1 * x0_i) / 2.0 +
                   dx_i * ((k1 + k2 * x0_i) / 3.0 +
                         dx_i * ((k2 + k3 * x0_i) / 4.0 +
                               dx_i * ((k3 + k4 * x0_i) / 5.0 +
                                     dx_i * ((k4 + k5 * x0_i) / 6.0 + dx_i * ((k5 + fdgd * x0_i) / 7.0 + dx_i * fdgd / 8.0)))))));
    }

    sdata[tid] = sum;
    __syncthreads();

    if (tid == 0) {
        for (unsigned i = 1; i < BLOCK_SIZE; ++i) {
            sum += sdata[i];
        }
#if __CUDA_ARCH__ < 600
        atomicAddCustom(results + blockIdx.y, sum);
#else
        atomicAdd(results + blockIdx.y, sum);
#endif
    }
}

extern "C" void spline_inner_product_gpu_m1(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results)
{
    constexpr unsigned int block_size = 256;
    dim3 threadBlock(block_size);
    dim3 blockGrid((max_num_points + threadBlock.x - 1) / threadBlock.x, num_splines);

    acc::zero(results, num_splines, stream);
    accLaunchKernel(spline_inner_product_gpu_kernel_m0<block_size>, blockGrid, threadBlock, 0, stream, num_points, offsets,
                    coeffs_1, coeffs_2, x0, dx, results);
}

template <unsigned int BLOCK_SIZE>
__global__ void spline_inner_product_gpu_kernel_m2(const int* __restrict__ num_points, const int* __restrict__ offsets,
                                                   const double4* __restrict__ coeffs_1,
                                                   const double4* __restrict__ coeffs_2, const double* __restrict__ x0,
                                                   const double* __restrict__ dx, double* __restrict__ results)
{
    static_assert(BLOCK_SIZE >= 64, "block size requirement for reduction not fulfilled!");
    assert(BLOCK_SIZE == blockDim.x);

    __shared__ double sdata[BLOCK_SIZE];
    const auto tid    = threadIdx.x;
    const auto i    = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    const auto offset = offsets[blockIdx.y];
    const auto N      = num_points[blockIdx.y];

    double sum = 0.0;
    if (i < N - 1) {
        const double4 f_coeff = coeffs_1[offset + i];
        const double4 g_coeff = coeffs_2[offset + i];
        const double dx_i     = dx[offset + i];
        const double x0_i     = x0[offset + i];

        const double k0 = f_coeff.x * g_coeff.x;
        const double k1 = f_coeff.w * g_coeff.y + f_coeff.z * g_coeff.z + f_coeff.y * g_coeff.w;
        const double k2 = f_coeff.w * g_coeff.x + f_coeff.z * g_coeff.y + f_coeff.y * g_coeff.z + f_coeff.x * g_coeff.w;
        const double k3 = f_coeff.z * g_coeff.x + f_coeff.y * g_coeff.y + f_coeff.x * g_coeff.z;
        const double k4 = f_coeff.w * g_coeff.z + f_coeff.z * g_coeff.w;
        const double k5 = f_coeff.y * g_coeff.x + f_coeff.x * g_coeff.y;
        const double k6 = f_coeff.w * g_coeff.w; // 25 OPS

        double v = dx_i * k6 * 0.11111111111111111111;

        const double r1 = k4 * 0.125 + k6 * x0_i * 0.25;
        v               = dx_i * (r1 + v);

        const double r2 = (k1 + x0_i * (2.0 * k4 + k6 * x0_i)) * 0.14285714285714285714;
        v               = dx_i * (r2 + v);

        const double r3 = (k2 + x0_i * (2.0 * k1 + k4 * x0_i)) * 0.16666666666666666667;
        v               = dx_i * (r3 + v);

        const double r4 = (k3 + x0_i * (2.0 * k2 + k1 * x0_i)) * 0.2;
        v               = dx_i * (r4 + v);

        const double r5 = (k5 + x0_i * (2.0 * k3 + k2 * x0_i)) * 0.25;
        v               = dx_i * (r5 + v);

        const double r6 = (k0 + x0_i * (2.0 * k5 + k3 * x0_i)) * 0.33333333333333333333;
        v               = dx_i * (r6 + v);

        const double r7 = (x0_i * (2.0 * k0 + x0_i * k5)) * 0.5;
        v               = dx_i * (r7 + v);

        sum += dx_i * (k0 * x0_i * x0_i + v);
    }

    sdata[tid] = sum;
    __syncthreads();

    // reduction
    // for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
    //     if (tid < s)
    //         sdata[tid] += sdata[tid + s];
    //     __syncthreads();
    // }

    // // single warp reduction -> no syncthreads necessary
    // if(tid < 32) {
    //     sdata[tid] += sdata[tid + 32];
    //     sdata[tid] += sdata[tid + 16];
    //     sdata[tid] += sdata[tid + 8];
    //     sdata[tid] += sdata[tid + 4];
    //     sdata[tid] += sdata[tid + 2];
    //     sdata[tid] += sdata[tid + 1];
    // }

    // if(tid == 0) {
    //  results[blockIdx.x] = sdata[0];
    // }
    if (tid == 0) {
        for (unsigned i = 1; i < BLOCK_SIZE; ++i) {
            sum += sdata[i];
        }
#if __CUDA_ARCH__ < 600
        atomicAddCustom(results + blockIdx.y, sum);
#else
        atomicAdd(results + blockIdx.y, sum);
#endif
    }
}

extern "C" void spline_inner_product_gpu_m2(acc_stream_t stream, const int num_splines, const int max_num_points,
                                            const int* num_points, const int* offsets, const double4* coeffs_1,
                                            const double4* coeffs_2, const double* x0, const double* dx,
                                            double* results)
{
    constexpr unsigned int block_size = 256;
    dim3 threadBlock(block_size);
    dim3 blockGrid((max_num_points + threadBlock.x - 1) / threadBlock.x, num_splines);

    acc::zero(results, num_splines, stream);
    accLaunchKernel(spline_inner_product_gpu_kernel_m0<block_size>, blockGrid, threadBlock, 0, stream, num_points,
                    offsets, coeffs_1, coeffs_2, x0, dx, results);
}

