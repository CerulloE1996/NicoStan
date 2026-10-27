#pragma once
#include "nico_auto_chunking.hpp"

namespace nico::math::auto_chunking {

//// ---- group operands gathered once per distinct group within each chunk ------------------------------------

template <typename T, bool keep_row_index = true>
struct gathered_at_unique_groups {
    const T& vector;
    const std::vector<int>& index;
};

template <typename T>
inline auto gathered_at_unique_groups_of(const T& vector,
                                         const std::vector<int>& index) {

    return gathered_at_unique_groups<T>{vector, index};

}

template <typename T>
inline auto gathered_at_unique_group_values_of(const T& vector,
                                               const std::vector<int>& index) {

    return gathered_at_unique_groups<T, false>{vector, index};

}

template <typename T>
struct is_unique_group_gather : std::false_type {};

template <typename T, bool keep_row_index>
struct is_unique_group_gather<gathered_at_unique_groups<T, keep_row_index>> : std::true_type {};

template <typename T>
struct values_at_unique_groups {
    T values;
    std::vector<int> source_index;
    std::vector<int> row_index;
};

template <typename T>
inline const auto& values_of_unique_groups(const values_at_unique_groups<T>& tile) {

    return tile.values;

}

template <typename T>
inline const auto& indices_of_unique_groups(const values_at_unique_groups<T>& tile) {

    return tile.row_index;

}

template <bool on_main_tape, typename T, bool keep_row_index>
inline auto unique_groups_for_the_tile(const gathered_at_unique_groups<T, keep_row_index>& shared,
                                       const int first,
                                       const int n_rows) {

    const int n_groups = static_cast<int>(shared.vector.rows());
    std::vector<int> source_index(shared.index.begin() + first, shared.index.begin() + first + n_rows);
    std::sort(source_index.begin(), source_index.end());
    source_index.erase(std::unique(source_index.begin(), source_index.end()), source_index.end());
    source_index.shrink_to_fit();
    if (!source_index.empty()) {
          stan::math::check_range("auto_chunking group", "group index", n_groups, source_index.front());
          stan::math::check_range("auto_chunking group", "group index", n_groups, source_index.back());
    }
    std::vector<int> row_index;
    if constexpr (keep_row_index) {
          row_index.resize(n_rows);
          for (int i = 0; i < n_rows; ++i) {
                row_index[i] = static_cast<int>(std::lower_bound(source_index.begin(), source_index.end(),
                                                                 shared.index[first + i]) - source_index.begin()) + 1;
          }
    }
    using source_scalar = stan::scalar_type_t<T>;
    using scalar = std::conditional_t<stan::is_var<source_scalar>::value, stan::math::var, double>;
    constexpr int columns = T::IsVectorAtCompileTime ? 1 : Eigen::Dynamic;
    using matrix = Eigen::Matrix<scalar, Eigen::Dynamic, columns>;
    matrix values(source_index.size(), shared.vector.cols());
    for (Eigen::Index j = 0; j < shared.vector.cols(); ++j) {
          for (std::size_t i = 0; i < source_index.size(); ++i) {
                values.coeffRef(i, j) = shared.vector.coeff(source_index[i] - 1, j);
          }
    }
    return values_at_unique_groups<matrix>{std::move(values), std::move(source_index), std::move(row_index)};

}


template <typename T>
inline const auto& vars_of_group_shared(const T& shared) {

    if constexpr (is_unique_group_gather<T>::value) return shared.vector;
    else return vars_of_shared(shared);

}

template <typename T>
inline auto scalar_type_of_group_shared(const T& shared) {

    return stan::return_type_t<std::decay_t<decltype(vars_of_group_shared(shared))>>();

}

template <typename T>
inline auto group_shared_for_the_tile(const T& shared,
                                      const int first,
                                      const int n_rows) {

    if constexpr (is_unique_group_gather<T>::value) {
          return unique_groups_for_the_tile<false>(shared, first, n_rows);
    } else {
          return shared_for_the_tile(shared, first, n_rows);
    }

}

template <typename T>
inline auto group_shared_on_the_main_tape(const T& shared,
                                          const int N) {

    if constexpr (is_unique_group_gather<T>::value) {
          return unique_groups_for_the_tile<true>(shared, 0, N);
    } else {
          return shared_on_the_main_tape(shared, N);
    }

}

template <typename T, typename Tile>
inline void add_adjoints_of_group_tile(double* partials_of_this_shared,
                                       const T& shared,
                                       const Tile& shared_of_the_tile,
                                       const int first,
                                       const int n_rows) {

    if constexpr (!stan::is_var<decltype(scalar_type_of_group_shared(shared))>::value) {
          return;
    } else if constexpr (is_unique_group_gather<T>::value) {

          const int count = static_cast<int>(shared_of_the_tile.source_index.size());
          for (Eigen::Index j = 0; j < shared.vector.cols(); ++j) {
                for (int i = 0; i < count; ++i) {
                      const Eigen::Index offset = shared_of_the_tile.source_index[i] - 1 +
                                                  j * shared.vector.rows();
                      const auto variable = shared_of_the_tile.values.coeff(i, j);
                      partials_of_this_shared[offset] += variable.adj();
                      variable.vi_->adj_ = 0.0;
                }
          }

    } else {
          add_adjoints_of_the_tile(partials_of_this_shared, shared, shared_of_the_tile, first, n_rows);
    }

}

template <std::size_t... I, typename Tile, typename... Shared>
inline void add_adjoints_of_group_shared(std::index_sequence<I...>,
                                         double* partials,
                                         const std::array<std::size_t, sizeof...(Shared)>& offsets,
                                         const Tile& tile,
                                         const int first,
                                         const int n_rows,
                                         const Shared&... shared) {

    (add_adjoints_of_group_tile(partials + offsets[I], shared, std::get<I>(tile), first, n_rows), ...);

}


//// ---- borrowed group vars keep their outer adjoints unchanged, including on an exception ------------------

struct adjoints_of_borrowed_groups {
    std::vector<stan::math::vari*> touched;
    std::vector<std::pair<stan::math::vari*, double>> saved;

    template <typename T, typename Tile>
    void preserve(const T& shared,
                   const Tile& tile) {

          if constexpr (is_unique_group_gather<T>::value &&
                        stan::is_var<decltype(scalar_type_of_group_shared(shared))>::value) {
                touched.reserve(touched.size() + tile.values.size());
                for (Eigen::Index j = 0; j < tile.values.cols(); ++j) {
                      for (Eigen::Index i = 0; i < tile.values.rows(); ++i) {
                            auto* variable = tile.values.coeff(i, j).vi_;
                            touched.push_back(variable);
                            if (variable->adj_ != 0.0) saved.emplace_back(variable, variable->adj_);
                            variable->adj_ = 0.0;
                      }
                }
          }

    }

    ~adjoints_of_borrowed_groups() {

          for (auto* variable : touched) variable->adj_ = 0.0;
          for (std::size_t i = saved.size(); i > 0; --i) {
                saved[i - 1].first->adj_ = saved[i - 1].second;
          }

    }
};

template <std::size_t... I, typename Tile, typename... Shared>
inline void preserve_adjoints_of_borrowed_groups(std::index_sequence<I...>,
                                                 adjoints_of_borrowed_groups& adjoints,
                                                 const Tile& tile,
                                                 const Shared&... shared) {

    (adjoints.preserve(shared, std::get<I>(tile)), ...);

}


//// ---- grouped partial arrays kept until Stan recovers the outer tape -------------------------------------

struct partials_of_grouped_rows : stan::math::chainable_alloc {
    std::vector<stan::math::vari*> varis;
    std::vector<double> partials;

    explicit partials_of_grouped_rows(const std::size_t size)
          : varis(size), partials(size, 0.0) {}
};

//// ---- the sum over rows 0 .. N - 1, chunk by chunk ----------------------------------------------------------

template <typename F, typename... Shared>
inline auto sum_of_grouped_rows_chunk_by_chunk(const std::vector<chunk_of_rows>& chunks,
                                       const F& sum_of_rows,
                                       const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_group_shared(shared))...>;
    //// (with no var among the shared arguments, as in log_prob<double>: the plain sum, chunk by chunk)
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {

          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {
                const int first = chunk.first;
                const int n_rows = chunk.n_rows;
                value += stan::math::apply([&](auto&&... of_the_tile) { return sum_of_rows(first, n_rows,
                                                                                            of_the_tile...); },
                                           std::make_tuple(group_shared_for_the_tile(shared, first, n_rows)...));
          }
          return value;

    } else {

          const std::array<std::size_t, sizeof...(Shared)> counts = {
                stan::math::count_vars(vars_of_group_shared(shared))...};
          std::array<std::size_t, sizeof...(Shared)> offsets{};
          std::size_t n_shared_vars = 0;
          for (std::size_t s = 0; s < counts.size(); ++s) {
                offsets[s] = n_shared_vars;
                n_shared_vars += counts[s];
          }
          auto& arena = stan::math::ChainableStack::instance_->memalloc_;
          stan::math::vari** varis;
          double* partials;
          if constexpr ((is_unique_group_gather<Shared>::value || ...)) {
                auto* storage = new partials_of_grouped_rows(n_shared_vars);
                varis = storage->varis.data();
                partials = storage->partials.data();
          } else {
                varis = arena.alloc_array<stan::math::vari*>(n_shared_vars);
                partials = arena.alloc_array<double>(n_shared_vars);
          }
          stan::math::vari** next_vari = varis;
          ((next_vari = save_varis_chunk_by_chunk(next_vari, vars_of_group_shared(shared))), ...);
          for (std::size_t first = 0; first < n_shared_vars; first += 4096) {
                std::fill_n(partials + first, std::min<std::size_t>(4096, n_shared_vars - first), 0.0);
          }
          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {

                const int first = chunk.first;
                const int n_rows = chunk.n_rows;
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(group_shared_for_the_tile(shared, first, n_rows)...);
                adjoints_of_borrowed_groups borrowed;
                preserve_adjoints_of_borrowed_groups(std::index_sequence_for<Shared...>{}, borrowed,
                                                      shared_of_the_tile, shared...);
                stan::math::var sum_of_tile = stan::math::apply(
                      [&](auto&&... of_the_tile) { return sum_of_rows(first, n_rows, of_the_tile...); },
                      shared_of_the_tile);
                sum_of_tile.grad();
                value += sum_of_tile.val();
                add_adjoints_of_group_shared(std::index_sequence_for<Shared...>{}, partials, offsets,
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

//// ---- a sliced vector has exactly the N rows of the sum -----------------------------------------------------

template <typename T>
inline void check_the_rows_of_group_shared(const T& shared,
                                     const int N) {

    if constexpr (is_unique_group_gather<T>::value) {
          stan::math::check_size_match("auto_chunking", "rows of the sum", N, "size of the group index",
                                       static_cast<int>(shared.index.size()));
    } else if constexpr (is_sliced_at_rows<T>::value) {
          stan::math::check_size_match("auto_chunking", "rows of the sum", N, "size of the sliced vector",
                                       static_cast<int>(shared.vector.size()));
    }

}

//// ---- the sum over rows 0 .. N - 1 in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking --

template <typename F, typename... Shared>
inline auto sum_of_grouped_rows_in_the_chunks_of_reduce_sum_static(const int N,
                                                           const int N_chunks_of_auto_chunking,
                                                           const F& sum_of_rows,
                                                           const Shared&... shared) {

    (check_the_rows_of_group_shared(shared, N), ...);
    const std::vector<chunk_of_rows> chunks
          = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_group_shared(shared))...>;
    if constexpr (stan::is_var<scalar_of_the_sum>::value) {
          //// (one chunk of all the rows is no chunking: the sum on the main tape, as stanc wrote it)
          if (chunks.size() == 1) {
                return stan::math::var(stan::math::apply(
                      [&](auto&&... on_the_main_tape) { return sum_of_rows(0, N, on_the_main_tape...); },
                      std::make_tuple(group_shared_on_the_main_tape(shared, N)...)));
          }
    }
    return sum_of_grouped_rows_chunk_by_chunk(chunks, sum_of_rows, shared...);

}

//// ---- the same, with stanc's own statement given (a statement whose N-row locals stay where stanc made them):
////      one chunk of all the rows runs that statement exactly as stanc wrote it, so that AUTO-CHUNKING at one
////      chunk is the model itself (no view of the rows, no copy of a shared vector)

template <typename G, typename F, typename... Shared>
inline auto sum_of_grouped_rows_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(
      const int N,
      const int N_chunks_of_auto_chunking,
      const G& as_stanc_wrote_it,
      const F& sum_of_rows,
      const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_group_shared(shared))...>;
    if constexpr (stan::is_var<scalar_of_the_sum>::value) {
          //// (one chunk: the chunk size is N or more)
          if (chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking) >= N) {
                return stan::math::var(as_stanc_wrote_it());
          }
    }
    return sum_of_grouped_rows_in_the_chunks_of_reduce_sum_static(N, N_chunks_of_auto_chunking, sum_of_rows, shared...);

}


}
