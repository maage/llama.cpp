#pragma once

// Block layout constants per ggml type. Not included by common.cuh on purpose: only the matrix multiplication
// kernels need them.

#include "common.cuh"

template <ggml_type type>
struct ggml_cuda_type_traits;

template<>
struct ggml_cuda_type_traits<GGML_TYPE_F16> {
    static constexpr int qk = 1;
    static constexpr int qr = 1;
    static constexpr int bs = sizeof(ggml_half);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q1_0> {
    static constexpr int qk = QK1_0;
    static constexpr int qr = QR1_0;
    static constexpr int qi = QI1_0;
    static constexpr int bs = sizeof(block_q1_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_0> {
    static constexpr int qk = QK2_0;
    static constexpr int qr = QR2_0;
    static constexpr int qi = QI2_0;
    static constexpr int bs = sizeof(block_q2_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_0> {
    static constexpr int qk = QK4_0;
    static constexpr int qr = QR4_0;
    static constexpr int qi = QI4_0;
    static constexpr int bs = sizeof(block_q4_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_1> {
    static constexpr int qk = QK4_1;
    static constexpr int qr = QR4_1;
    static constexpr int qi = QI4_1;
    static constexpr int bs = sizeof(block_q4_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_0> {
    static constexpr int qk = QK5_0;
    static constexpr int qr = QR5_0;
    static constexpr int qi = QI5_0;
    static constexpr int bs = sizeof(block_q5_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_1> {
    static constexpr int qk = QK5_1;
    static constexpr int qr = QR5_1;
    static constexpr int qi = QI5_1;
    static constexpr int bs = sizeof(block_q5_1);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q8_0> {
    static constexpr int qk = QK8_0;
    static constexpr int qr = QR8_0;
    static constexpr int qi = QI8_0;
    static constexpr int bs = sizeof(block_q8_0);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_MXFP4> {
    static constexpr int qk = QK_MXFP4;
    static constexpr int qr = QR_MXFP4;
    static constexpr int qi = QI_MXFP4;
    static constexpr int bs = sizeof(block_mxfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
    static constexpr int bs = sizeof(block_nvfp4);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
    static constexpr int bs = sizeof(block_q2_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
    static constexpr int bs = sizeof(block_q3_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
    static constexpr int bs = sizeof(block_q4_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q5_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR5_K;
    static constexpr int qi = QI5_K;
    static constexpr int bs = sizeof(block_q5_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
    static constexpr int bs = sizeof(block_q6_K);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XXS;
    static constexpr int qi = QI2_XXS;
    static constexpr int bs = sizeof(block_iq2_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_XS;
    static constexpr int qi = QI2_XS;
    static constexpr int bs = sizeof(block_iq2_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ2_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_S;
    static constexpr int qi = QI2_S;
    static constexpr int bs = sizeof(block_iq2_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_XXS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_XXS;
    static constexpr int qi = QI3_XXS;
    static constexpr int bs = sizeof(block_iq3_xxs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_S;
    static constexpr int qi = QI1_S;
    static constexpr int bs = sizeof(block_iq1_s);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ1_M> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR1_M;
    static constexpr int qi = QI1_M;
    static constexpr int bs = sizeof(block_iq1_m);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_NL> {
    static constexpr int qk = QK4_NL;
    static constexpr int qr = QR4_NL;
    static constexpr int qi = QI4_NL;
    static constexpr int bs = sizeof(block_iq4_nl);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_XS;
    static constexpr int qi = QI4_XS;
    static constexpr int bs = sizeof(block_iq4_xs);
};

template<>
struct ggml_cuda_type_traits<GGML_TYPE_IQ3_S> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_S;
    static constexpr int qi = QI3_S;
    static constexpr int bs = sizeof(block_iq3_s);
};
