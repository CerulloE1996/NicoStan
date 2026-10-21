//// log_Phi_and_inv_Phi_from_log_p_over_an_Eigen_vector.hpp
////
//// ---- (moved here from BayesMVP's src/MVP_functions/MVP_log_scale_grad_calc_fns.hpp, lines 40-90, unchanged;
////      included there at the same place, after the kernels and double_fns.hpp)
#pragma once
//// ---- two small dispatch helpers (chunked SIMD over a contiguous Eigen vector, scalar remainder)
namespace nico::math {

inline Eigen::Matrix<double, -1, 1> fn_EIGEN_log_Phi(const Eigen::Matrix<double, -1, 1> &x,
                                                      const std::string &vect_type) {

      const int N = x.size();
      Eigen::Matrix<double, -1, 1> out(N);
      int i = 0;
#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
      if (vect_type == "AVX512") {
          for (; i + 8 <= N; i += 8) {
              _mm512_storeu_pd(out.data() + i, nico_log_Phi_AVX512(_mm512_loadu_pd(x.data() + i)));
          }
      }
#endif
#if defined(__AVX2__)
      if (vect_type == "AVX2") {
          for (; i + 4 <= N; i += 4) {
              _mm256_storeu_pd(out.data() + i, nico_log_Phi_AVX2(_mm256_loadu_pd(x.data() + i)));
          }
      }
#endif
      for (; i < N; ++i) out(i) = nico_log_Phi(x(i));
      return out;

}

}  // namespace nico::math

using nico::math::fn_EIGEN_log_Phi;

namespace nico::math {

inline Eigen::Matrix<double, -1, 1> fn_EIGEN_inv_Phi_from_log_p(const Eigen::Matrix<double, -1, 1> &log_p,
                                                                 const Eigen::Matrix<double, -1, 1> &log_1m_p,
                                                                 const std::string &vect_type) {

      const int N = log_p.size();
      Eigen::Matrix<double, -1, 1> out(N);
      int i = 0;
#if defined(__AVX512VL__) && defined(__AVX512F__) && defined(__AVX512DQ__)
      if (vect_type == "AVX512") {
          for (; i + 8 <= N; i += 8) {
              _mm512_storeu_pd(out.data() + i,
                               nico_inv_Phi_from_log_p_wo_checks_AVX512(_mm512_loadu_pd(log_p.data() + i),
                                                                        _mm512_loadu_pd(log_1m_p.data() + i)));
          }
      }
#endif
#if defined(__AVX2__)
      if (vect_type == "AVX2") {
          for (; i + 4 <= N; i += 4) {
              _mm256_storeu_pd(out.data() + i,
                               nico_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p.data() + i),
                                                                      _mm256_loadu_pd(log_1m_p.data() + i)));
          }
      }
#endif
      for (; i < N; ++i) out(i) = nico_inv_Phi_from_log_p(log_p(i), log_1m_p(i));
      return out;

}

}  // namespace nico::math

using nico::math::fn_EIGEN_inv_Phi_from_log_p;






















