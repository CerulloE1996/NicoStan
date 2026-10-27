#pragma once
////
//// nico_auto_chunking_scan.hpp: dot_product(a, x) - weighted_sum_of_log_cumulative_sum_exp(w, x, log_offset),
//// the partial likelihood of a Cox model, done tile by tile on stan::math::var, after nico_auto_chunking.hpp.
//// With L_k = log(exp(log_offset) + exp(x_1) + ... + exp(x_k)), its value is sum_k a_k x_k - sum_k w_k L_k; the
//// running sum crosses the tiles, so:
////   pass 1 (values only, the tiles in order): x of each tile's rows, from the values of the shared parameters;
////   the log of the running sum before the tile, and the tile's shift (the largest of it and of the tile's x);
////   pass 2 (the tiles in reverse order, each on a nested tape): x of the tile's rows, from copies of the shared
////   parameters; the tile's running sums started from the sum before it (a var of the tile), each expensive call
////   its own autodiff operation (nico_exp, cumulative_sum, nico_log), as in the model's own function; the
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
//// ---- dot_product(a, x) - weighted_sum_of_log_cumulative_sum_exp(w, x, log_offset), tile by tile -----------
////
template <typename T_a, typename T_w, typename F, typename G, typename... Shared>
inline auto dot_product_minus_weighted_log_cumulative_sum_exp_in_tiles(const int N,
                                                                       const int rows_per_tile,
                                                                       const T_a& a,
                                                                       const T_w& w,
                                                                       const double log_offset,
                                                                       const F& x_of_rows,
                                                                       const G& as_stanc_wrote_it,
                                                                       std::ostream* pstream__,
                                                                       const Shared&... shared) {

    using scalar_of_the_value = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    const double minus_infinity = -std::numeric_limits<double>::infinity();
    const int n_tiles = (N + rows_per_tile - 1) / rows_per_tile;
    ////
    //// ---- pass 1: the log of the running sum before each tile, and each tile's shift (values only) -----------
    ////
    std::vector<double> log_sum_before(n_tiles);
    std::vector<double> shift(n_tiles);
    double log_sum = log_offset;
    double value = 0.0;
    bool underflows = false;
    for (int t = 0; t < n_tiles; ++t) {

          const int first = t * rows_per_tile;
          const int n_rows = std::min(rows_per_tile, N - first);
          const Eigen::VectorXd x = stan::math::apply(
                [&](auto&&... of_the_tile) { return Eigen::VectorXd(x_of_rows(first, n_rows, of_the_tile...)); },
                std::make_tuple(values_for_the_tile(shared, first, n_rows)...));
          log_sum_before[t] = log_sum;
          shift[t] = std::max(log_sum, x.maxCoeff());
          const Eigen::VectorXd terms = nico_exp(Eigen::VectorXd((x.array() - shift[t]).matrix()), pstream__);
          Eigen::VectorXd running = stan::math::cumulative_sum(terms);
          running.array() += (log_sum == minus_infinity) ? 0.0 : nico_exp(log_sum - shift[t], pstream__);
          if (running.coeff(0) < 1e-280) {
                underflows = true;
                break;
          }
          const Eigen::VectorXd log_running = nico_log(running, pstream__);
          value += rows_of(a, first, n_rows).dot(x) - shift[t] * rows_of(w, first, n_rows).sum()
                   - rows_of(w, first, n_rows).dot(log_running);
          log_sum = shift[t] + nico_log(running.coeff(n_rows - 1), pstream__);

    }
    if (underflows) return scalar_of_the_value(as_stanc_wrote_it());
    if constexpr (!stan::is_var<scalar_of_the_value>::value) {

          return value;

    } else {

          ////
          //// ---- pass 2: the tiles in reverse order, each on a nested tape --------------------------------------
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

                const int first = t * rows_per_tile;
                const int n_rows = std::min(rows_per_tile, N - first);
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                const stan::math::var sum_before(log_sum_before[t] == minus_infinity
                                                 ? 0.0 : nico_exp(log_sum_before[t] - shift[t], pstream__));
                const auto x = stan::math::apply(
                      [&](auto&&... of_the_tile) { return x_of_rows(first, n_rows, of_the_tile...); },
                      shared_of_the_tile);
                const auto terms = nico_exp(stan::math::subtract(x, shift[t]), pstream__);
                const auto running = stan::math::add(stan::math::cumulative_sum(terms), sum_before);
                const stan::math::var value_of_the_tile = stan::math::dot_product(rows_of(a, first, n_rows), x)
                      - shift[t] * rows_of(w, first, n_rows).sum()
                      - stan::math::dot_product(rows_of(w, first, n_rows), nico_log(running, pstream__));
                const stan::math::var ending_log_sum
                      = shift[t] + nico_log(stan::math::add(sum_before, stan::math::sum(terms)), pstream__);
                value_of_the_tile.vi_->adj_ = 1.0;
                ending_log_sum.vi_->adj_ = adjoint_of_the_next_starting_log_sum;
                stan::math::grad();
                value_of_the_tiles += value_of_the_tile.val();
                adjoint_of_the_next_starting_log_sum = sum_before.adj() * sum_before.val();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_tile, first, n_rows, shared...);

          }
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

}  // namespace nico::math::auto_chunking
