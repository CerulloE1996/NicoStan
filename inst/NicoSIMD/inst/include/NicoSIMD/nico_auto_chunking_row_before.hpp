#pragma once
//// nico_auto_chunking_row_before.hpp: AUTO-CHUNKING of a sum over the rows whose term of a row also reads the
//// row before it (the contracts of auto_chunking_contracts_of_row_sums_with_the_row_before.hpp), after
//// nico_auto_chunking.hpp in the model's user header.
//// The chunks are exactly reduce_sum_static's (chunks_of_reduce_sum_static); a chunk whose first row is not
//// row 0 and does not start a group (group_start not 1 there) is run on the row before it and its own rows, the
//// row before put first (row_before = true), as a row whose own term is zero (its data set by the converter to
//// its contract's values), so that the chunk's first term reads it; the adjoints of that row go back to its
//// parameters with the chunk's other adjoints. One chunk of all the rows is stanc's own statement.
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

namespace nico::math::auto_chunking {

//// ---- whether the chunk whose first row is first (0-based) is given the row before it ----------------------

template <typename T_group_start>
inline bool needs_the_row_before(const T_group_start& group_start,
                                 const int first) {

    if (first == 0) return false;
    return static_cast<double>(element_of(group_start, first)) != 1.0;

}

//// ---- the rows first .. first + n_rows - 1 of an array of the data, the first set to value if it is the row
////      before -----------------------------------------------------------------------------------------------

template <typename T, typename T_value>
inline std::vector<T> array_rows_with_the_row_before_set_to(const std::vector<T>& x,
                                                            const int first,
                                                            const int n_rows,
                                                            const bool row_before,
                                                            const T_value value) {

    std::vector<T> rows(x.begin() + first, x.begin() + first + n_rows);
    if (row_before) rows[0] = static_cast<T>(value);
    return rows;

}

//// ---- the same for a vector of the data ----------------------------------------------------------------------

template <typename T, typename T_value, stan::require_eigen_col_vector_t<T>* = nullptr>
inline Eigen::VectorXd rows_with_the_row_before_set_to(const T& x,
                                                       const int first,
                                                       const int n_rows,
                                                       const bool row_before,
                                                       const T_value value) {

    Eigen::VectorXd rows = stan::math::value_of(x.segment(first, n_rows));
    if (row_before) rows.coeffRef(0) = static_cast<double>(value);
    return rows;

}

//// ---- the sum over rows 0 .. N - 1, chunk by chunk, each chunk with the row before it where it needs it ------

template <typename T_group_start, typename F, typename... Shared>
inline auto sum_of_rows_with_the_row_before_chunk_by_chunk(const std::vector<chunk_of_rows>& chunks,
                                                           const T_group_start& group_start,
                                                           const F& sum_of_rows,
                                                           const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    //// (with no var among the shared arguments, as in log_prob<double>: the plain sum, chunk by chunk)
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {

          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {
                const bool row_before = needs_the_row_before(group_start, chunk.first);
                const int first = chunk.first - (row_before ? 1 : 0);
                const int n_rows = chunk.n_rows + (row_before ? 1 : 0);
                value += stan::math::apply([&](auto&&... of_the_tile) {
                                                 return sum_of_rows(first, n_rows, row_before, of_the_tile...);
                                           },
                                           std::make_tuple(shared_for_the_tile(shared, first, n_rows)...));
          }
          return value;

    } else {

          const std::array<std::size_t, sizeof...(Shared)> counts = {
                stan::math::count_vars(vars_of_shared(shared))...};
          std::array<std::size_t, sizeof...(Shared)> offsets{};
          std::size_t n_shared_vars = 0;
          for (std::size_t s = 0; s < counts.size(); ++s) {
                offsets[s] = n_shared_vars;
                n_shared_vars += counts[s];
          }
          auto& arena = stan::math::ChainableStack::instance_->memalloc_;
          stan::math::vari** varis = arena.alloc_array<stan::math::vari*>(n_shared_vars);
          double* partials = arena.alloc_array<double>(n_shared_vars);
          stan::math::vari** next_vari = varis;
          ((next_vari = save_varis_chunk_by_chunk(next_vari, vars_of_shared(shared))), ...);
          for (std::size_t first = 0; first < n_shared_vars; first += 4096) {
                std::fill_n(partials + first, std::min<std::size_t>(4096, n_shared_vars - first), 0.0);
          }
          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {

                //// (the chunk's rows, and the row before them when the first does not start a group: the tile's
                ////  rows, read and given their adjoints back at those rows)
                const bool row_before = needs_the_row_before(group_start, chunk.first);
                const int first = chunk.first - (row_before ? 1 : 0);
                const int n_rows = chunk.n_rows + (row_before ? 1 : 0);
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                stan::math::var sum_of_tile = stan::math::apply(
                      [&](auto&&... of_the_tile) {
                            return sum_of_rows(first, n_rows, row_before, of_the_tile...);
                      },
                      shared_of_the_tile);
                sum_of_tile.grad();
                value += sum_of_tile.val();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_tile, first, n_rows, shared...);

          }
          //// (the result's backward pass: its adjoint times the partials, added to the shared vars chunk by
          ////  chunk, 4096 at a time)
          return stan::math::make_callback_var(value, [varis, partials, n_shared_vars](auto& result) {

                const double adjoint = result.adj_;
                for (std::size_t first = 0; first < n_shared_vars; first += 4096) {
                      const std::size_t last = std::min<std::size_t>(first + 4096, n_shared_vars);
                      for (std::size_t i = first; i < last; ++i) varis[i]->adj_ += adjoint * partials[i];
                }

          });

    }

}

//// ---- the sum over rows 0 .. N - 1 in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking ----

template <typename T_group_start, typename F, typename... Shared>
inline auto sum_of_rows_with_the_row_before_in_the_chunks_of_reduce_sum_static(const int N,
                                                                              const int N_chunks_of_auto_chunking,
                                                                              const T_group_start& group_start,
                                                                              const F& sum_of_rows,
                                                                              const Shared&... shared) {

    (check_the_rows_of_shared(shared, N), ...);
    stan::math::check_size_match("auto_chunking", "rows of the sum", N, "size of the group starts",
                                 static_cast<int>(group_start.size()));
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    if constexpr (stan::is_var<scalar_of_the_sum>::value) {
          //// (one chunk of all the rows is no chunking: the sum on the main tape, with no row before)
          if (chunks.size() == 1) {
                return stan::math::var(stan::math::apply(
                      [&](auto&&... on_the_main_tape) { return sum_of_rows(0, N, false, on_the_main_tape...); },
                      std::make_tuple(shared_on_the_main_tape(shared, N)...)));
          }
    }
    return sum_of_rows_with_the_row_before_chunk_by_chunk(chunks, group_start, sum_of_rows, shared...);

}

//// ---- the same, with stanc's own statement given: one chunk of all the rows runs it exactly as stanc wrote it

template <typename T_group_start, typename G, typename F, typename... Shared>
inline auto sum_of_rows_with_the_row_before_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(
      const int N,
      const int N_chunks_of_auto_chunking,
      const T_group_start& group_start,
      const G& as_stanc_wrote_it,
      const F& sum_of_rows,
      const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    if constexpr (stan::is_var<scalar_of_the_sum>::value) {
          //// (one chunk: the chunk size is N or more)
          if (chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking) >= N) {
                return stan::math::var(as_stanc_wrote_it());
          }
    }
    return sum_of_rows_with_the_row_before_in_the_chunks_of_reduce_sum_static(N, N_chunks_of_auto_chunking,
                                                                              group_start, sum_of_rows,
                                                                              shared...);

}

}  // namespace nico::math::auto_chunking
