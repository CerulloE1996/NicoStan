#pragma once

//// ---- AUTO-CHUNKING of a direct call of a partial-sum function that reads NicoStan's nuisance block itself ------
////
//// As nico_auto_chunking_partial_sum_calls.hpp, for a nico_reduce_sum partial-sum function whose rows read their
//// own values of NicoStan's nuisance block (stan_external_nuisance_values_of_the_block_owned_by_NicoStan.hpp). Each
//// tile is one chunk of the nuisance block, as each chunk of nico_reduce_sum is: its own terms first .. last, the
//// values it reads made nodes of the tile's own tape, and their adjoints collected after the tile's gradient; the
//// node of the sum adds the collected adjoints, times its own adjoint, into NicoStan's nuisance gradient
//// (nicostan_precomputed_gradients_vari_of_the_nuisance_block, the node nico_reduce_sum returns). Needs the
//// nuisance-block runtime of NicoStan's reduce_sum (options NicoStan.reduce_sum_nuisance_block = TRUE).

#include <stan/math/rev.hpp>

#include <utility>
#include <vector>

namespace nico::math::auto_chunking {

template <typename F, typename... Shared>
inline auto sum_of_rows_chunk_by_chunk_reading_the_nuisance_block(const std::vector<chunk_of_rows>& chunks,
                                                                  const F& sum_of_rows,
                                                                  const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    //// (with no var among the shared arguments, as in log_prob<double>: the readers give numbers)
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {

          return sum_of_rows_chunk_by_chunk(chunks, sum_of_rows, shared...);

    } else {

          namespace internal = stan::math::internal;
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
          //// (the caller's nuisance block: null outside NicoStan's gradient call, where the readers give values
          ////  whose gradient is not kept)
          internal::nicostan_nuisance_block_t* block = internal::nicostan_nuisance_block;
          std::vector<std::pair<int, double>> nuisance_derivatives;
          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {

                const int first = chunk.first;
                const int n_rows = chunk.n_rows;
                const stan::math::nested_rev_autodiff tile;
                internal::nicostan_scope_of_a_chunk_of_the_nuisance_block nuisance_scope(block, first,
                                                                                        first + n_rows - 1);
                internal::nicostan_scope_of_the_nuisance_derivative_collection derivative_scope(nuisance_derivatives);
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                stan::math::var sum_of_tile = stan::math::apply(
                      [&](auto&&... of_the_tile) { return sum_of_rows(first, n_rows, of_the_tile...); },
                      shared_of_the_tile);
                sum_of_tile.grad();
                internal::nicostan_add_the_adjoints_of_the_chunk_of_the_nuisance_block();
                value += sum_of_tile.val();
                add_adjoints_of_every_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
                                             shared_of_the_tile, first, n_rows, shared...);

          }
          return stan::math::var(new internal::nicostan_precomputed_gradients_vari_of_the_nuisance_block(
                value, n_shared_vars, varis, partials, block, nuisance_derivatives));

    }

}

//// ---- the sum over rows 0 .. N - 1 in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking --

template <typename F, typename... Shared>
inline auto sum_of_rows_in_the_chunks_of_reduce_sum_static_reading_the_nuisance_block(
      const int N,
      const int N_chunks_of_auto_chunking,
      const F& sum_of_rows,
      const Shared&... shared) {

    (check_the_rows_of_shared(shared, N), ...);
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    if constexpr (stan::is_var<scalar_of_the_sum>::value) {
          //// (one chunk of all the rows is no chunking: the sum on the main tape, as stanc wrote it)
          if (chunks.size() == 1) {
                return stan::math::var(stan::math::apply(
                      [&](auto&&... on_the_main_tape) { return sum_of_rows(0, N, on_the_main_tape...); },
                      std::make_tuple(shared_on_the_main_tape(shared, N)...)));
          }
    }
    return sum_of_rows_chunk_by_chunk_reading_the_nuisance_block(chunks, sum_of_rows, shared...);

}

//// ---- the same, with stanc's own call given: one chunk runs that call exactly as stanc wrote it ------------

template <typename G, typename F, typename... Shared>
inline auto sum_of_rows_in_the_chunks_of_reduce_sum_static_reading_the_nuisance_block_or_as_stanc_wrote_it(
      const int N,
      const int N_chunks_of_auto_chunking,
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
    return sum_of_rows_in_the_chunks_of_reduce_sum_static_reading_the_nuisance_block(N, N_chunks_of_auto_chunking,
                                                                                     sum_of_rows, shared...);

}

}  // namespace nico::math::auto_chunking
