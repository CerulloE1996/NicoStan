#pragma once
////
//// nico_auto_chunking_scan_with_a_lean_first_pass.hpp: dot_product(a, x) -
//// weighted_sum_of_log_cumulative_sum_exp(w, x, log_offset), the partial likelihood of a Cox model, done tile
//// by tile on stan::math::var, after
//// nico_auto_chunking.hpp. With L_k = log(exp(log_offset) + exp(x_1) + ... + exp(x_k)), its value is
//// sum_k a_k x_k - sum_k w_k L_k; the running sum crosses the tiles, so:
////   pass 1 (values only, the tiles in order): x of each tile's rows, from the values of the shared parameters;
////   the log of the running sum before the tile, and the tile's shift (the largest of it and of the tile's x),
////   from the sum of the tile's terms alone (no running sums, no logs of them); with one tile, no pass 1;
////   pass 2 (the tiles in reverse order, each on a nested tape): x of the tile's rows, from copies of the shared
////   parameters; the tile's running sums started from the sum before it (a var of the tile), each expensive call
////   its own autodiff operation (exp, cumulative_sum, log: Stan's in a Stan::math cell, NicoSIMD's nico_exp and
////   nico_log in a nico::math cell, as the model's own function calls them); the
////   backward pass seeded with 1 on the tile's value and, on the tile's ending log sum, with the adjoint of the
////   next tile's starting one; the starting var's adjoint then passes to the tile before.
//// The shift of each tile is a constant: the value does not depend on it. Where a tile's first running sum is
//// below 1e-280 (pass 1), the statement is done as stanc wrote it (as_stanc_wrote_it), unchunked: the model's
//// own function then computes those positions again, as it does unchunked.
////

namespace nico::math::auto_chunking {

////
//// ---- the values of a shared argument at a tile's rows (for pass 1) ------------------------------------------
////
template <typename T>
inline auto values_for_the_tile(const T& shared,
                                const int first,
                                const int n_rows) {

    if constexpr (is_gathered_at_rows<T>::value) {
          //// (the tile's indices checked at once, as Stan's z[group] checks them)
          const Eigen::Map<const Eigen::VectorXi> index_of_the_tile(shared.index.data() + first, n_rows);
          const int size_of_the_vector = static_cast<int>(shared.vector.size());
          if ((n_rows > 0) && ((index_of_the_tile.minCoeff() < 1) ||
                               (index_of_the_tile.maxCoeff() > size_of_the_vector))) {
                const int outside = (index_of_the_tile.minCoeff() < 1) ? index_of_the_tile.minCoeff()
                                                                        : index_of_the_tile.maxCoeff();
                stan::math::check_range("vector[multi] indexing", "index", size_of_the_vector, outside);
          }
          return Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(n_rows, [&](const Eigen::Index i) {
                return stan::math::value_of(shared.vector.coeff(shared.index[first + i] - 1));
          }));
    } else {
          return stan::math::eval(stan::math::value_of(shared));
    }

}

////
//// ---- the maths of the cell: Stan's exp and log in a Stan::math cell, NicoSIMD's in a nico::math cell -------
////
struct maths_of_Stan {

    template <typename T>
    static auto exp(const T& x,
                    std::ostream* /*pstream__*/) {

        return stan::math::eval(stan::math::exp(x));

    }
    template <typename T>
    static auto log(const T& x,
                    std::ostream* /*pstream__*/) {

        return stan::math::eval(stan::math::log(x));

    }

};

struct maths_of_NicoSIMD {

    template <typename T>
    static auto exp(const T& x,
                    std::ostream* pstream__) {

        return nico_exp(x, pstream__);

    }
    template <typename T>
    static auto log(const T& x,
                    std::ostream* pstream__) {

        return nico_log(x, pstream__);

    }

};

////
//// ---- dot_product(a, x) - weighted_sum_of_log_cumulative_sum_exp(w, x, log_offset), tile by tile -----------
////
template <typename Maths, typename T_a, typename T_w, typename F, typename G, typename... Shared>
inline auto dot_product_minus_weighted_log_cumulative_sum_exp_chunk_by_chunk(
        const std::vector<chunk_of_rows>& chunks,
        const T_a& a,
        const T_w& w,
        const double log_offset,
        const F& x_of_rows,
        const G& as_stanc_wrote_it,
        std::ostream* pstream__,
        const Shared&... shared) {

    using scalar_of_the_value = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    constexpr bool with_vars = stan::is_var<scalar_of_the_value>::value;
    const double minus_infinity = -std::numeric_limits<double>::infinity();
    const int n_tiles = static_cast<int>(chunks.size());
    ////
    //// ---- pass 1: the log of the running sum before each tile, and each tile's shift (values only) -----------
    ////
    //// (with vars, only what pass 2 needs: each tile's shift and carried log sum, from the sum of its terms;
    ////  with one tile, nothing: pass 2 takes the shift from the tile's own x. Without vars, the value too)
    std::vector<double> log_sum_before(n_tiles);
    std::vector<double> shift(n_tiles);
    double log_sum = log_offset;
    double value = 0.0;
    bool underflows = false;
    const int n_tiles_of_pass_1 = (with_vars && (n_tiles == 1)) ? 0 : n_tiles;
    for (int t = 0; t < n_tiles_of_pass_1; ++t) {

          const int first = chunks[t].first;
          const int n_rows = chunks[t].n_rows;
          const Eigen::VectorXd x = stan::math::apply(
                [&](auto&&... of_the_tile) { return Eigen::VectorXd(x_of_rows(first, n_rows, of_the_tile...)); },
                std::make_tuple(values_for_the_tile(shared, first, n_rows)...));
          log_sum_before[t] = log_sum;
          shift[t] = std::max(log_sum, x.maxCoeff());
          const Eigen::VectorXd terms = Maths::exp(Eigen::VectorXd((x.array() - shift[t]).matrix()), pstream__);
          const double sum_before = (log_sum == minus_infinity) ? 0.0 : Maths::exp(log_sum - shift[t], pstream__);
          if (terms.coeff(0) + sum_before < 1e-280) {
                underflows = true;
                break;
          }
          if constexpr (with_vars) {
                log_sum = shift[t] + Maths::log(sum_before + terms.sum(), pstream__);
          } else {
                Eigen::VectorXd running = stan::math::cumulative_sum(terms);
                running.array() += sum_before;
                const Eigen::VectorXd log_running = Maths::log(running, pstream__);
                value += rows_of(a, first, n_rows).dot(x) - shift[t] * rows_of(w, first, n_rows).sum()
                         - rows_of(w, first, n_rows).dot(log_running);
                log_sum = shift[t] + Maths::log(running.coeff(n_rows - 1), pstream__);
          }

    }
    if (underflows) return scalar_of_the_value(as_stanc_wrote_it());
    if constexpr (!with_vars) {

          return value;

    } else {

          ////
          //// ---- pass 2: the tiles in reverse order, each on a nested tape ------------------------------------
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
          double value_of_the_tiles = 0.0;
          double adjoint_of_the_next_starting_log_sum = 0.0;
          for (int t = n_tiles - 1; t >= 0; --t) {

                const int first = chunks[t].first;
                const int n_rows = chunks[t].n_rows;
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                const auto x = stan::math::apply(
                      [&](auto&&... of_the_tile) { return x_of_rows(first, n_rows, of_the_tile...); },
                      shared_of_the_tile);
                //// (one tile: no pass 1; its shift and its rare case from its own x)
                if (n_tiles_of_pass_1 == 0) {
                      log_sum_before[t] = log_offset;
                      shift[t] = std::max(log_offset, stan::math::value_of(x).maxCoeff());
                      const double first_term
                            = Maths::exp(stan::math::value_of(x).coeff(0) - shift[t], pstream__);
                      const double sum_before_value = (log_offset == minus_infinity)
                                                      ? 0.0 : Maths::exp(log_offset - shift[t], pstream__);
                      if (first_term + sum_before_value < 1e-280) {
                            underflows = true;
                            break;
                      }
                }
                const stan::math::var sum_before(log_sum_before[t] == minus_infinity
                                                 ? 0.0 : Maths::exp(log_sum_before[t] - shift[t], pstream__));
                const auto terms = Maths::exp(stan::math::subtract(x, shift[t]), pstream__);
                const auto running = stan::math::add(stan::math::cumulative_sum(terms), sum_before);
                const stan::math::var value_of_the_tile = stan::math::dot_product(rows_of(a, first, n_rows), x)
                      - shift[t] * rows_of(w, first, n_rows).sum()
                      - stan::math::dot_product(rows_of(w, first, n_rows), Maths::log(running, pstream__));
                const stan::math::var ending_log_sum
                      = shift[t] + Maths::log(stan::math::add(sum_before, stan::math::sum(terms)), pstream__);
                value_of_the_tile.vi_->adj_ = 1.0;
                ending_log_sum.vi_->adj_ = adjoint_of_the_next_starting_log_sum;
                stan::math::grad();
                value_of_the_tiles += value_of_the_tile.val();
                adjoint_of_the_next_starting_log_sum = sum_before.adj() * sum_before.val();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_tile, first, n_rows, shared...);

          }
          if (underflows) return scalar_of_the_value(as_stanc_wrote_it());
          //// (the result's backward pass: its adjoint times the partials, added to the shared vars chunk by
          ////  chunk, 4096 at a time)
          return stan::math::make_callback_var(value_of_the_tiles,
                                               [varis, partials, n_shared_vars](auto& result) {

                const double adjoint = result.adj_;
                for (std::size_t first = 0; first < n_shared_vars; first += 4096) {
                      const std::size_t last = std::min<std::size_t>(first + 4096, n_shared_vars);
                      for (std::size_t i = first; i < last; ++i) varis[i]->adj_ += adjoint * partials[i];
                }

          });

    }

}

////
//// ---- the same in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking -----------------------
////
template <typename Maths, typename T_a, typename T_w, typename F, typename G, typename... Shared>
inline auto dot_product_minus_weighted_log_cumulative_sum_exp_in_the_chunks_of_reduce_sum_static(
        const int N,
        const int N_chunks_of_auto_chunking,
        const T_a& a,
        const T_w& w,
        const double log_offset,
        const F& x_of_rows,
        const G& as_stanc_wrote_it,
        std::ostream* pstream__,
        const Shared&... shared) {

    using scalar_of_the_value = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    //// (one chunk of all the rows is no chunking: the statement as stanc wrote it, on the main tape)
    if (chunks.size() == 1) return scalar_of_the_value(as_stanc_wrote_it());
    return dot_product_minus_weighted_log_cumulative_sum_exp_chunk_by_chunk<Maths>(
          chunks, a, w, log_offset, x_of_rows, as_stanc_wrote_it, pstream__, shared...);

}

////
//// ---- dot_product(a, x) - nico_dot_product_log_cumulative_sum_exp(w, x, log_offset), chunk by chunk ---------
////
//// (a fused nico::math cell: on each chunk, nico_dot_product_log_cumulative_sum_exp(w, x, log_sum_before) and
////  nico_log_sum_exp of log_sum_before and x, both NicoSIMD's general fused functions, one autodiff node each,
////  with log_sum_before the log of the running sum before the chunk; pass 1 gives each chunk its
////  log_sum_before (values only), pass 2 the chunks in reverse order on nested tapes, the adjoint of each
////  chunk's ending log sum carried to the chunk after it; the function's own scan keeps its rare case, so there
////  is no fallback)
template <typename T_a, typename T_w, typename F, typename... Shared>
inline auto dot_product_minus_nico_dot_product_log_cumulative_sum_exp_chunk_by_chunk(
        const std::vector<chunk_of_rows>& chunks,
        const T_a& a,
        const T_w& w,
        const double log_offset,
        const F& x_of_rows,
        std::ostream* pstream__,
        const Shared&... shared) {

    using scalar_of_the_value = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    constexpr bool with_vars = stan::is_var<scalar_of_the_value>::value;
    const int n_chunks = static_cast<int>(chunks.size());
    ////
    //// ---- pass 1: the log of the running sum before each chunk (values only; without vars, the value too) ----
    ////
    std::vector<double> log_sum_before(n_chunks);
    double log_sum = log_offset;
    double value = 0.0;
    for (int t = 0; t < n_chunks; ++t) {

          const int first = chunks[t].first;
          const int n_rows = chunks[t].n_rows;
          const Eigen::VectorXd x = stan::math::apply(
                [&](auto&&... of_the_chunk) {
                      return Eigen::VectorXd(x_of_rows(first, n_rows, of_the_chunk...));
                },
                std::make_tuple(values_for_the_tile(shared, first, n_rows)...));
          log_sum_before[t] = log_sum;
          if constexpr (!with_vars) {
                value += rows_of(a, first, n_rows).dot(x)
                         - nico_dot_product_log_cumulative_sum_exp(Eigen::VectorXd(rows_of(w, first, n_rows)), x,
                                                                   log_sum, pstream__);
          }
          //// (of the values of w's type, so that NicoSIMD's function is looked up only where the cell calls it)
          stan::plain_type_t<decltype(rows_of(w, first, n_rows))> log_sum_and_x(n_rows + 1);
          log_sum_and_x << log_sum, x;
          log_sum = nico_log_sum_exp(log_sum_and_x, pstream__);

    }
    if constexpr (!with_vars) {

          return value;

    } else {

          ////
          //// ---- pass 2: the chunks in reverse order, each on a nested tape -----------------------------------
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
          double value_of_the_chunks = 0.0;
          double adjoint_of_the_next_starting_log_sum = 0.0;
          for (int t = n_chunks - 1; t >= 0; --t) {

                const int first = chunks[t].first;
                const int n_rows = chunks[t].n_rows;
                const stan::math::nested_rev_autodiff chunk;
                auto shared_of_the_chunk = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                const Eigen::Matrix<stan::math::var, -1, 1> x = stan::math::apply(
                      [&](auto&&... of_the_chunk) { return stan::math::eval(x_of_rows(first, n_rows,
                                                                                       of_the_chunk...)); },
                      shared_of_the_chunk);
                const stan::math::var starting_log_sum(log_sum_before[t]);
                const stan::math::var value_of_the_chunk
                      = stan::math::dot_product(rows_of(a, first, n_rows), x)
                        - nico_dot_product_log_cumulative_sum_exp(Eigen::VectorXd(rows_of(w, first, n_rows)), x,
                                                                  starting_log_sum, pstream__);
                Eigen::Matrix<scalar_of_the_value, -1, 1> starting_log_sum_and_x(n_rows + 1);
                starting_log_sum_and_x << starting_log_sum, x;
                const stan::math::var ending_log_sum = nico_log_sum_exp(starting_log_sum_and_x, pstream__);
                value_of_the_chunk.vi_->adj_ = 1.0;
                ending_log_sum.vi_->adj_ = adjoint_of_the_next_starting_log_sum;
                stan::math::grad();
                value_of_the_chunks += value_of_the_chunk.val();
                adjoint_of_the_next_starting_log_sum = starting_log_sum.adj();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_chunk, first, n_rows, shared...);

          }
          //// (the result's backward pass: its adjoint times the partials, added to the shared vars chunk by
          ////  chunk, 4096 at a time)
          return stan::math::make_callback_var(value_of_the_chunks,
                                               [varis, partials, n_shared_vars](auto& result) {

                const double adjoint = result.adj_;
                for (std::size_t first = 0; first < n_shared_vars; first += 4096) {
                      const std::size_t last = std::min<std::size_t>(first + 4096, n_shared_vars);
                      for (std::size_t i = first; i < last; ++i) varis[i]->adj_ += adjoint * partials[i];
                }

          });

    }

}

////
//// ---- the same in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking -----------------------
////
template <typename T_a, typename T_w, typename F, typename G, typename... Shared>
inline auto dot_product_minus_nico_dot_product_log_cumulative_sum_exp_in_the_chunks_of_reduce_sum_static(
        const int N,
        const int N_chunks_of_auto_chunking,
        const T_a& a,
        const T_w& w,
        const double log_offset,
        const F& x_of_rows,
        const G& as_stanc_wrote_it,
        std::ostream* pstream__,
        const Shared&... shared) {

    using scalar_of_the_value = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    //// (one chunk of all the rows is no chunking: the statement as stanc wrote it, on the main tape)
    if (chunks.size() == 1) return scalar_of_the_value(as_stanc_wrote_it());
    return dot_product_minus_nico_dot_product_log_cumulative_sum_exp_chunk_by_chunk(
          chunks, a, w, log_offset, x_of_rows, pstream__, shared...);

}

}  // namespace nico::math::auto_chunking
