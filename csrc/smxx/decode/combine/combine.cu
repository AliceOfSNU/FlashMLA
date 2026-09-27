#include "combine.h"

#include <type_traits>
#include <cstdlib>
#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <kerutils/kerutils.cuh>

#include "params.h"
#include "utils.h"

using namespace cute;

namespace smxx::decode {

template<typename ElementT, int HEAD_DIM_V, int BLOCK_SIZE_M, int MAX_SPLITS, int NUM_THREADS, int N_DCHUNK>
__global__ void __launch_bounds__(NUM_THREADS)
flash_fwd_mla_combine_kernel(__grid_constant__ const CombineParams params) {
    // grid_shape: [batch_size*s_q, N_DCHUNK, h_q/BLOCK_SIZE_M]
    // Each CTA gathers the activation of some heads from one batch, do scaling & accumulation, and save the result.
    // blockIdx.y (N_DCHUNK-way) partitions the HEAD_DIM_V output across CTAs so that at
    // low batch -- where grid.x*grid.z is far below the SM count -- the merge is spread
    // over more CTAs to fill idle SMs. Each CTA owns a 512/N_DCHUNK slice of every head's
    // output; the per-split lse scales are head-only (D-independent) so they are just
    // recomputed per chunk (cheap). No cross-CTA reduction, single launch. N_DCHUNK==1 is
    // byte-for-byte the upstream kernel.
    static_assert(NUM_THREADS/32 == BLOCK_SIZE_M); // The number of warps == block_size_m
    const int batch_s_q_idx = blockIdx.x;
    const int batch_idx = batch_s_q_idx / params.s_q;
    const int s_q_idx = batch_s_q_idx - batch_idx * params.s_q;
    const int h_block_idx = blockIdx.z;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    constexpr int HEAD_DIM_PER_CTA = HEAD_DIM_V / N_DCHUNK;
    const int dchunk = blockIdx.y;                 // which head_dim slice this CTA owns
    const int d_offset = dchunk * HEAD_DIM_PER_CTA; // element offset into the (unit-stride) head_dim

    int num_valid_heads = std::min(BLOCK_SIZE_M, params.h_q - BLOCK_SIZE_M*h_block_idx);
    if (warp_idx >= num_valid_heads) {
        return;
    }

    const int start_split_idx = __ldg(params.num_splits_ptr + batch_idx);
    const int end_split_idx = __ldg(params.num_splits_ptr + batch_idx + 1);
    const int my_num_splits = end_split_idx - start_split_idx;
    if (my_num_splits == 1) {
        return;
    }
    
    FLASH_DEVICE_ASSERT(my_num_splits <= MAX_SPLITS);
    
    Tensor gLseAccum = make_tensor(
        make_gmem_ptr((float*)params.lse_accum + start_split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + h_block_idx*BLOCK_SIZE_M),
        Shape<Int<MAX_SPLITS>, Int<BLOCK_SIZE_M>>{},
        make_stride(params.stride_lse_accum_split, _1{})
    );
    Tensor gLse = make_tensor(
        make_gmem_ptr((float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + h_block_idx*BLOCK_SIZE_M),
        Shape<Int<BLOCK_SIZE_M>>{},
        Stride<_1>{}
    );
    
    __shared__ float smem_buf[BLOCK_SIZE_M][MAX_SPLITS];

    // Wait for the previous kernel (the MLA kernel) to finish
    cudaGridDependencySynchronize();

    // Prefetch
    static_assert(HEAD_DIM_PER_CTA % (32*4) == 0);
    constexpr int ELEMS_PER_THREAD = HEAD_DIM_PER_CTA / (32*4);
    float* oaccum_ptr = params.o_accum + start_split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + (h_block_idx*BLOCK_SIZE_M + warp_idx)*params.stride_o_accum_h_q + d_offset;
    float4 datas[ELEMS_PER_THREAD];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        datas[i] = *(float4*)(oaccum_ptr + lane_idx*4 + i*128); // NOTE We don't use __ldg here since it is incompatible with PDL
    }

    // Warp #i gathers LseAccum for seq #i
    {
        constexpr int NUM_LSE_PER_THREAD = cute::ceil_div(MAX_SPLITS, 32);
        float local_lse[NUM_LSE_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            local_lse[i] = split_idx < my_num_splits ? gLseAccum(split_idx, warp_idx) : -INFINITY;
        }

        float max_lse = -INFINITY;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            max_lse = max(max_lse, local_lse[i]);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i)
            sum_lse = sum_lse + exp2f(local_lse[i] - max_lse);
        CUTLASS_PRAGMA_UNROLL
        for (int offset = 16; offset >= 1; offset /= 2)
            sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse == -INFINITY) ? INFINITY : log2f(sum_lse) + max_lse;
        if (lane_idx == 0 && dchunk == 0)  // gLse is head-only; only one D-chunk need write it
            gLse(warp_idx) = global_lse / (float)M_LOG2E;
        
        if (params.attn_sink != nullptr) {
            int q_head_idx = h_block_idx*BLOCK_SIZE_M + warp_idx;
            float attn_sink = __ldg(params.attn_sink + q_head_idx);
            if (global_lse != INFINITY) {
                // If attn_sink is +inf, global_lse will be +inf and scale factors will be exp2f(local_lse - inf) = 0 (since local_lse never becomes +inf)
                // If attn_sink is -inf, this has no effect on global_lse
                global_lse += log2f(1 + exp2f(attn_sink*CUDART_L2E_F - global_lse));
            } else {
                // We have no tokens to attend, so global lse should be attn_sink*CUDART_L2E_F (+inf if it's -inf or +inf)
                global_lse = attn_sink == -INFINITY ? +INFINITY : attn_sink*CUDART_L2E_F;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < NUM_LSE_PER_THREAD; ++i) {
            const int split_idx = i*32 + lane_idx;
            smem_buf[warp_idx][split_idx] = exp2f(local_lse[i] - global_lse);
        }
    }

    __syncwarp();

    // Warp #i accumulates activation for seq #i
    {
        float4 result[ELEMS_PER_THREAD];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i)
            result[i] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 1
        for (int split = 0; split < my_num_splits; ++split) {
            float lse_scale = smem_buf[warp_idx][split];
            // if (lse_scale != 0.f) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                result[i].x += lse_scale * datas[i].x;
                result[i].y += lse_scale * datas[i].y;
                result[i].z += lse_scale * datas[i].z;
                result[i].w += lse_scale * datas[i].w;
                if (split != my_num_splits-1) {
                    datas[i] = *(float4*)(oaccum_ptr + (split+1)*params.stride_o_accum_split + lane_idx*4 + i*128);
                }
            }
            // }
        }
        
        const int h_q_idx = h_block_idx*BLOCK_SIZE_M + warp_idx;
        ElementT* o_ptr = (ElementT*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + h_q_idx*params.stride_o_h_q + d_offset;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float4 data = result[i];
            ElementT data_converted[4];
            data_converted[0] = (ElementT)(data.x);
            data_converted[1] = (ElementT)(data.y);
            data_converted[2] = (ElementT)(data.z);
            data_converted[3] = (ElementT)(data.w);
            static_assert(sizeof(ElementT) == 2);
            *(uint64_t*)(o_ptr + lane_idx*4 + i*128) = *(uint64_t*)data_converted;
        }
    }
}


#define MLA_NUM_SPLITS_SWITCH(NUM_SPLITS, NAME, ...)       \
    [&] {                                                  \
        if (NUM_SPLITS <= 32) {                            \
            constexpr static int NAME = 32;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 64) {                     \
            constexpr static int NAME = 64;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 96) {                     \
            constexpr static int NAME = 96;                \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 128) {                    \
            constexpr static int NAME = 128;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 160) {                    \
            constexpr static int NAME = 160;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 192) {                    \
            constexpr static int NAME = 192;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 224) {                    \
            constexpr static int NAME = 224;               \
            return __VA_ARGS__();                          \
        } else if (NUM_SPLITS <= 256) {                    \
            constexpr static int NAME = 256;               \
            return __VA_ARGS__();                          \
        } else {                                           \
            FLASH_ASSERT(false);                           \
        }                                                  \
    }()


template<typename ElementT>
void run_flash_mla_combine_kernel(CombineParams &params) {
    static constexpr int HEAD_DIM_V = 512;  // Since only this head dimension is supported by Flash MLA
    FLASH_ASSERT(params.d_v == HEAD_DIM_V);

    // Fan the merge over N_DCHUNK head_dim slices to fill idle SMs at low batch. The
    // upstream grid.x*grid.z = (b*s_q)*ceil(h_q/8) is far below the SM count at low batch,
    // leaving most SMs idle while each CTA serially loops all splits. Pick N_DCHUNK in
    // {1,2,4} (512/N_DCHUNK must stay a multiple of 128 for the float4 path) so the grid
    // just reaches the SM count; N_DCHUNK==1 (high batch) is exactly the upstream launch.
    static const int num_sms = [] {
        int dev = 0, n = 1;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }();
    const int base_grid = params.b * params.s_q * ku::ceil_div(params.h_q, 8);
    int ndchunk = 1;
    if (base_grid > 0) {
        const int want = ku::ceil_div(num_sms, base_grid);
        ndchunk = want >= 4 ? 4 : (want >= 2 ? 2 : 1);
    }
    // Override for A/B benchmarking / safety: FLASH_MLA_COMBINE_NDCHUNK=1 forces the
    // exact upstream single-launch behavior; 2 or 4 pins the fan-out. Unset/0 = auto.
    static const int forced_ndchunk = [] {
        const char* e = getenv("FLASH_MLA_COMBINE_NDCHUNK");
        return e ? atoi(e) : 0;
    }();
    if (forced_ndchunk == 1 || forced_ndchunk == 2 || forced_ndchunk == 4)
        ndchunk = forced_ndchunk;

    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, NUM_SPLITS, [&] {
        constexpr int BLOCK_SIZE_M = 8;
        constexpr int NUM_THREADS = BLOCK_SIZE_M*32;
        constexpr size_t smem_size = BLOCK_SIZE_M*(NUM_SPLITS+1)*sizeof(float);
        auto launch = [&](auto ndchunk_ic) {
            constexpr int N_DCHUNK = decltype(ndchunk_ic)::value;
            auto combine_kernel = &flash_fwd_mla_combine_kernel<ElementT, HEAD_DIM_V, BLOCK_SIZE_M, NUM_SPLITS, NUM_THREADS, N_DCHUNK>;
            CHECK_CUDA(cudaFuncSetAttribute(combine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
            cudaLaunchAttribute attribute[1];
            attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t combine_kernel_config = {
                dim3(params.b * params.s_q, N_DCHUNK, ku::ceil_div(params.h_q, BLOCK_SIZE_M)),
                dim3(NUM_THREADS, 1, 1),
                0,
                params.stream,
                attribute,
                1
            };
            CHECK_CUDA(cudaLaunchKernelEx(&combine_kernel_config, combine_kernel, params));
        };
        if (ndchunk == 4) launch(std::integral_constant<int, 4>{});
        else if (ndchunk == 2) launch(std::integral_constant<int, 2>{});
        else launch(std::integral_constant<int, 1>{});
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_flash_mla_combine_kernel<cutlass::bfloat16_t>(CombineParams &params);

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_mla_combine_kernel<cutlass::half_t>(CombineParams &params);
#endif

}
