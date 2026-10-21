//// log_Phi_and_inv_Phi_from_log_p_over_an_Eigen_vector_templated_by_the_SIMD_level.hpp
////
//// ---- (moved here from BayesMVP's src/MVP_functions/MVP_log_scale_grad_calc_fns_T.hpp, lines 130-201, unchanged;
////      included there at the same place, after the kernels and double_fns.hpp)
#pragma once
namespace nico::math {

template <Vec vec>
inline void fn_apply_log_Phi_exact_inplace_T(Eigen::Matrix<double, -1, 1> &values_in_log_Phi_out) {
  
        double *data_pointer = values_in_log_Phi_out.data();
        const int n_values = static_cast<int>(values_in_log_Phi_out.size());
        int index_value = 0;
        if constexpr (vec == Vec::AVX512) {
          #if BMVP_HAS_AVX512
            for (; index_value + 8 <= n_values; index_value += 8) {
              _mm512_storeu_pd(data_pointer + index_value,
                               nico_log_Phi_AVX512(_mm512_loadu_pd(data_pointer + index_value)));
            }
          #elif BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(data_pointer + index_value,
                               nico_log_Phi_AVX2(_mm256_loadu_pd(data_pointer + index_value)));
            }
          #endif
        } else if constexpr (vec == Vec::AVX2) {
          #if BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(data_pointer + index_value,
                               nico_log_Phi_AVX2(_mm256_loadu_pd(data_pointer + index_value)));
            }
          #endif
        }
        for (; index_value < n_values; ++index_value) {
          data_pointer[index_value] = nico_log_Phi(data_pointer[index_value]);
        }
  
}

}  // namespace nico::math

using nico::math::fn_apply_log_Phi_exact_inplace_T;




namespace nico::math {

template <Vec vec>
inline void fn_inv_Phi_from_log_p_exact_T(  const Eigen::Matrix<double, -1, 1> &log_p_vec,
                                            const Eigen::Matrix<double, -1, 1> &log_1m_p_vec,
                                            Eigen::Matrix<double, -1, 1> &Z_out_vec
) {
  
        const int n_values = static_cast<int>(log_p_vec.size());
        Z_out_vec.resize(n_values);
        const double *log_p_pointer = log_p_vec.data();
        const double *log_1m_p_pointer = log_1m_p_vec.data();
        double *Z_out_pointer = Z_out_vec.data();
        int index_value = 0;
        if constexpr (vec == Vec::AVX512) {
          #if BMVP_HAS_AVX512
            for (; index_value + 8 <= n_values; index_value += 8) {
              _mm512_storeu_pd(Z_out_pointer + index_value,
                               nico_inv_Phi_from_log_p_wo_checks_AVX512(_mm512_loadu_pd(log_p_pointer + index_value),
                                                                        _mm512_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #elif BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(Z_out_pointer + index_value,
                               nico_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p_pointer + index_value),
                                                                      _mm256_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #endif
        } else if constexpr (vec == Vec::AVX2) {
          #if BMVP_HAS_AVX2
            for (; index_value + 4 <= n_values; index_value += 4) {
              _mm256_storeu_pd(Z_out_pointer + index_value,
                               nico_inv_Phi_from_log_p_wo_checks_AVX2(_mm256_loadu_pd(log_p_pointer + index_value),
                                                                      _mm256_loadu_pd(log_1m_p_pointer + index_value)));
            }
          #endif
        }
        for (; index_value < n_values; ++index_value) {
          Z_out_pointer[index_value] = nico_inv_Phi_from_log_p(log_p_pointer[index_value],
                        log_1m_p_pointer[index_value]);
        }
  
}

}  // namespace nico::math

using nico::math::fn_inv_Phi_from_log_p_exact_T;






















