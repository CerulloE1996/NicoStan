#pragma once
//// nico_auto_chunking_running_sums.hpp: AUTO-CHUNKING of a statement whose rows meet a running sum
//// (cumulative_sum) and the largest of a row-wise vector (max), after nico_auto_chunking.hpp. Its value is a sum
//// over the rows once each chunk's running sums start from the totals of the chunks before it (the carries):
////   the largest row: the rows' values chunk by chunk (no tape); the row at the largest value is then made on
////   the main tape, from the shared parameters themselves (the same var Stan's max returns);
////   pass 1 (values only, the chunks in order): each chunk from the carries of the chunks before it, which gives
////   every chunk's starting carries, and the model's own test of its first row;
////   pass 2 (the chunks in reverse order, each on a nested tape): the chunk from vars of its starting carries;
////   its backward pass seeded with 1 on its value and, on its ending carries, with the adjoints of the next
////   chunk's starting carries; the adjoints of the starting carries then pass to the chunk before.
//// The chunks are exactly reduce_sum_static's (chunk_size_of_auto_chunking = ceiling(N / N_chunks)). Where the
//// model's own test of its first row fails (pass 1), the statement is done as stanc wrote it, unchunked.
#include "nico_auto_chunking.hpp"

namespace nico::math::auto_chunking {

//// ---- one chunk is no chunking --------------------------------------------------------------------------------

inline bool is_one_chunk(const int N,
                         const int N_chunks_of_auto_chunking) {

    return chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking) >= N;

}

//// ---- the values of a shared argument at a chunk's rows, with no var made --------------------------------------

template <typename T>
inline auto values_of_shared_for_the_tile(const T& shared,
                                          const int first,
                                          const int n_rows) {

    if constexpr (is_sliced_at_rows<T>::value) {
          return Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(n_rows, [&](const Eigen::Index i) {
                return stan::math::value_of(element_of(shared.vector, first + i));
          }));
    } else if constexpr (is_matrix_gather<T>::value) {
          const Eigen::Map<const Eigen::VectorXi> index(shared.index.data() + first, n_rows);
          const int n_groups = static_cast<int>(shared.vector.rows());
          if ((n_rows > 0) && ((index.minCoeff() < 1) || (index.maxCoeff() > n_groups))) {
                const int outside = index.minCoeff() < 1 ? index.minCoeff() : index.maxCoeff();
                stan::math::check_range("matrix[multi] indexing", "index", n_groups, outside);
          }
          return Eigen::MatrixXd(Eigen::MatrixXd::NullaryExpr(n_rows, shared.vector.cols(),
                [&](const Eigen::Index i, const Eigen::Index j) {
                      return stan::math::value_of(shared.vector.coeff(shared.index[first + i] - 1, j));
                }));
    } else if constexpr (is_gathered_at_rows<T>::value) {
          const Eigen::Map<const Eigen::VectorXi> index(shared.index.data() + first, n_rows);
          const int size_of_the_vector = static_cast<int>(shared.vector.size());
          if ((n_rows > 0) && ((index.minCoeff() < 1) || (index.maxCoeff() > size_of_the_vector))) {
                const int outside = (index.minCoeff() < 1) ? index.minCoeff() : index.maxCoeff();
                stan::math::check_range("vector[multi] indexing", "index", size_of_the_vector, outside);
          }
          return Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(n_rows, [&](const Eigen::Index i) {
                return stan::math::value_of(shared.vector.coeff(shared.index[first + i] - 1));
          }));
    } else {
          return stan::math::eval(stan::math::value_of(shared));
    }

}

//// ---- a shared argument at one row k, on the main tape: its own vars, no copy ----------------------------------

template <typename T>
inline auto shared_at_one_row_on_the_main_tape(const T& shared,
                                               const int k) {

    if constexpr (is_sliced_at_rows<T>::value) {
          using scalar = std::decay_t<decltype(element_of(shared.vector, k))>;
          Eigen::Matrix<scalar, -1, 1> row(1);
          row(0) = element_of(shared.vector, k);
          return row;
    } else if constexpr (is_matrix_gather<T>::value) {
          const int n_groups = static_cast<int>(shared.vector.rows());
          stan::math::check_range("matrix[multi] indexing", "index", n_groups, shared.index[k]);
          using scalar = stan::scalar_type_t<std::decay_t<decltype(shared.vector)>>;
          return Eigen::Matrix<scalar, -1, -1>(shared.vector.row(shared.index[k] - 1));
    } else if constexpr (is_gathered_at_rows<T>::value) {
          const int size_of_the_vector = static_cast<int>(shared.vector.size());
          stan::math::check_range("vector[multi] indexing", "index", size_of_the_vector, shared.index[k]);
          using scalar = std::decay_t<decltype(shared.vector.coeff(0))>;
          Eigen::Matrix<scalar, -1, 1> row(1);
          row(0) = shared.vector.coeff(shared.index[k] - 1);
          return row;
    } else {
          return shared;
    }

}

//// ---- the largest of a row-wise vector, as Stan's max: the value at its first largest row --------------------
////      (rows_of_the_chunk(first, n_rows, shared...) gives the chunk's rows of the vector)

template <typename F, typename... Shared>
inline auto largest_row_in_the_chunks_of_reduce_sum_static(const int N,
                                                           const int N_chunks_of_auto_chunking,
                                                           const F& rows_of_the_chunk,
                                                           const Shared&... shared) {

    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    double largest = -std::numeric_limits<double>::infinity();
    int row_of_the_largest = -1;
    for (const chunk_of_rows& chunk : chunks) {

          const Eigen::VectorXd values = stan::math::value_of(stan::math::apply(
                [&](auto&&... of_the_tile) { return rows_of_the_chunk(chunk.first, chunk.n_rows, of_the_tile...); },
                std::make_tuple(values_of_shared_for_the_tile(shared, chunk.first, chunk.n_rows)...)));
          Eigen::Index i = 0;
          const double largest_of_the_chunk = values.maxCoeff(&i);
          if ((row_of_the_largest < 0) || (largest_of_the_chunk > largest)) {
                largest = largest_of_the_chunk;
                row_of_the_largest = chunk.first + static_cast<int>(i);
          }

    }
    using scalar_of_the_rows = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    if constexpr (!stan::is_var<scalar_of_the_rows>::value) {
          return largest;
    } else {
          const auto row = stan::math::apply(
                [&](auto&&... at_the_row) { return rows_of_the_chunk(row_of_the_largest, 1, at_the_row...); },
                std::make_tuple(shared_at_one_row_on_the_main_tape(shared, row_of_the_largest)...));
          return stan::math::var(row.coeff(0));
    }

}

//// ---- the sum over the rows of a statement with running sums, chunk by chunk with carries ----------------------
////      (rows_of_the_chunk(first, n_rows, carries_in, shared...) gives the tuple (the chunk's value, its ending
////       carries, whether its first row passed the model's own test), for doubles and for vars)

template <typename G, typename F, typename... Shared>
inline auto sum_of_rows_with_running_sums_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(
      const int N,
      const int N_chunks_of_auto_chunking,
      const int n_running_sums,
      const G& as_stanc_wrote_it,
      const F& rows_of_the_chunk,
      const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    using result_type = std::conditional_t<stan::is_var<scalar_of_the_sum>::value, stan::math::var, double>;
    if (is_one_chunk(N, N_chunks_of_auto_chunking)) return result_type(as_stanc_wrote_it());
    (check_the_rows_of_shared(shared, N), ...);
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    const int n_chunks = static_cast<int>(chunks.size());
    ////
    //// ---- pass 1: the values, the chunks in order: each chunk's starting carries -----------------------------------
    ////
    std::vector<Eigen::VectorXd> carries_in_of_the_chunk(n_chunks);
    Eigen::VectorXd carries = Eigen::VectorXd::Zero(n_running_sums);
    double value_of_the_chunks = 0.0;
    for (int c = 0; c < n_chunks; ++c) {

          carries_in_of_the_chunk[c] = carries;
          const auto of_the_chunk = stan::math::apply(
                [&](auto&&... of_the_tile) {
                      return rows_of_the_chunk(chunks[c].first, chunks[c].n_rows, carries_in_of_the_chunk[c],
                                               of_the_tile...);
                },
                std::make_tuple(values_of_shared_for_the_tile(shared, chunks[c].first, chunks[c].n_rows)...));
          if (!std::get<2>(of_the_chunk)) return result_type(as_stanc_wrote_it());
          value_of_the_chunks += stan::math::value_of(std::get<0>(of_the_chunk));
          carries = stan::math::value_of(std::get<1>(of_the_chunk));

    }
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {

          return value_of_the_chunks;

    } else {

          ////
          //// ---- pass 2: the chunks in reverse order, each on a nested tape ---------------------------------------
          ////
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
          Eigen::VectorXd adjoints_of_the_next_carries = Eigen::VectorXd::Zero(n_running_sums);
          for (int c = n_chunks - 1; c >= 0; --c) {

                const int first = chunks[c].first;
                const int n_rows = chunks[c].n_rows;
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                Eigen::Matrix<stan::math::var, -1, 1> carries_in(n_running_sums);
                for (int k = 0; k < n_running_sums; ++k) carries_in(k) = carries_in_of_the_chunk[c](k);
                const auto of_the_chunk = stan::math::apply(
                      [&](auto&&... of_the_tile) { return rows_of_the_chunk(first, n_rows, carries_in,
                                                                            of_the_tile...); },
                      shared_of_the_tile);
                const stan::math::var value_of_the_chunk = std::get<0>(of_the_chunk);
                const Eigen::Matrix<stan::math::var, -1, 1> carries_out = std::get<1>(of_the_chunk);
                value_of_the_chunk.vi_->adj_ += 1.0;
                for (int k = 0; k < n_running_sums; ++k) {
                      carries_out(k).vi_->adj_ += adjoints_of_the_next_carries(k);
                }
                stan::math::grad();
                value += value_of_the_chunk.val();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_tile, first, n_rows, shared...);
                for (int k = 0; k < n_running_sums; ++k) adjoints_of_the_next_carries(k) = carries_in(k).adj();

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

}  // namespace nico::math::auto_chunking
