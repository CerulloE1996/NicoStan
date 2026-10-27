#pragma once

//// ---- AUTO-CHUNKING of a density with no declared row-sum contract ------------------------------------------
////
//// A density whose arguments are all its rows or scalars is summed over the chunks reduce_sum_static makes, as a
//// univariate density is (nico_auto_chunking.hpp). Its first call on a var with more than one chunk also runs
//// stanc's own statement, on a nested tape whose memory is freed at once, and stops if the chunks' sum is not the
//// statement's value: a density that is not a sum over its rows never gives the log density of its chunks.
//// After nico_auto_chunking.hpp in the model's user header.

#include <stan/math/rev.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <string>

namespace nico::math::auto_chunking {

template <typename G, typename F, typename... Shared>
inline auto sum_of_rows_of_a_density_without_a_contract_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(
      const char* name_of_the_density,
      const int N,
      const int N_chunks_of_auto_chunking,
      const G& as_stanc_wrote_it,
      const F& sum_of_rows,
      const Shared&... shared) {

    auto value = sum_of_rows_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(N, N_chunks_of_auto_chunking,
                                                                                    as_stanc_wrote_it, sum_of_rows,
                                                                                    shared...);
    if constexpr (stan::is_var<std::decay_t<decltype(value)>>::value) {

          //// (once per call site and process: the first call with more than one chunk and finite values)
          static std::atomic<bool> checked{false};
          if (!checked.load(std::memory_order_relaxed)
              && (chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking) < N)) {

                double whole = 0.0;
                {
                      const stan::math::nested_rev_autodiff nested;
                      whole = stan::math::value_of(as_stanc_wrote_it());
                }
                const double chunked = stan::math::value_of(value);
                if (std::isfinite(whole) && std::isfinite(chunked)) {

                      const double scale = std::max({1.0, std::fabs(whole), std::fabs(chunked)});
                      if (std::fabs(chunked - whole) > 1e-8 * scale) {
                            throw std::runtime_error(std::string("AUTO-CHUNKING: ") + name_of_the_density
                                                     + " is not a sum over its rows (its chunks give "
                                                     + std::to_string(chunked) + ", stanc's own statement "
                                                     + std::to_string(whole)
                                                     + "): build the model with auto_chunking = FALSE");
                      }
                      checked.store(true, std::memory_order_relaxed);

                }

          }

    }
    return value;

}

}  // namespace nico::math::auto_chunking
