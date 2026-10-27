#pragma once
//// nico_auto_chunking.hpp: the sum over the rows of a model's data, done tile by tile on stan::math::var, with
//// no reduce_sum and no change to the model's own statement.
//// sum_of_rows_in_the_chunks_of_reduce_sum_static(N, N_chunks_of_auto_chunking, sum_of_rows, shared...): the
//// rows cut into exactly the chunks reduce_sum_static makes with chunk_size_of_auto_chunking = ceiling(N /
//// N_chunks_of_auto_chunking) (N_chunks_of_auto_chunking from the model's data, searched like N_chunks); for
//// each chunk of rows (first, n_rows), a nested tape with copies of the shared parameters, sum_of_rows(first,
//// n_rows, copies...) on it, its backward
//// pass at once, and the adjoints of the copies added to the partials; the result is one node with those
//// partials, so the sum may appear anywhere in the statement (its own adjoint multiplies them in the outer
//// backward pass). Each tile's memory is freed when the tile ends.
//// Inside a tile, the N-row data are read through rows_of(), with no copy; the tile's vectors can be Stan's own
//// struct-of-arrays var vectors (var_value<Eigen::VectorXd>), and the nico_ elementwise functions below take
//// them: NicoSIMD's kernel on the values array in place, and one reverse callback with NicoSIMD's derivative,
//// 64 values at a time.
#include <stan/math.hpp>
//// (in a model's user header, NicoSIMD's own user header comes first and has included it already)
#if __has_include("NicoSIMD/stan_external_functions.hpp")
#include "NicoSIMD/stan_external_functions.hpp"
#endif

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nico::math::auto_chunking {

using vector_soa = stan::math::var_value<Eigen::VectorXd>;

//// ---- the rows first .. first + n_rows - 1 of the data, with no copy -----------------------------------------

template <typename T, stan::require_eigen_col_vector_t<T>* = nullptr>
inline auto rows_of(const T& x,
                    const int first,
                    const int n_rows) {

    return x.segment(first, n_rows);

}

template <typename T, stan::require_eigen_matrix_dynamic_t<T>* = nullptr>
inline auto rows_of(const T& x,
                    const int first,
                    const int n_rows) {

    return x.middleRows(first, n_rows);

}

inline auto rows_of(const std::vector<double>& x,
                    const int first,
                    const int n_rows) {

    return Eigen::Map<const Eigen::VectorXd>(x.data() + first, n_rows);

}

//// (an array of integers, as Stan's to_vector() of it: the rows' values as doubles)
inline auto rows_of(const std::vector<int>& x,
                    const int first,
                    const int n_rows) {

    return Eigen::Map<const Eigen::VectorXi>(x.data() + first, n_rows).cast<double>();

}

//// (an array of the data given to a density, as an array of the rows' own values: integers stay integers)
template <typename T>
inline std::vector<T> array_rows_of(const std::vector<T>& x,
                                    const int first,
                                    const int n_rows) {

    return std::vector<T>(x.begin() + first, x.begin() + first + n_rows);

}

//// ---- rows first .. first + n_rows - 1 of a slice x[lower:upper] (1-based) of the data, with no copy ---------

//// (the slice checked to have the N rows of the sum, and to lie inside x, as Stan's x[lower:upper] checks it)
template <typename T>
inline void check_the_slice(const T& x,
                            const int lower,
                            const int upper,
                            const int N) {

    stan::math::check_size_match("auto_chunking", "rows of the slice", upper - lower + 1, "rows of the sum", N);
    if (N > 0) {
          stan::math::check_range("auto_chunking slice", "lower index", static_cast<int>(x.size()), lower);
          stan::math::check_range("auto_chunking slice", "upper index", static_cast<int>(x.size()), upper);
    }

}

template <typename T>
inline auto rows_of_slice(const T& x,
                          const int lower,
                          const int upper,
                          const int N,
                          const int first,
                          const int n_rows) {

    check_the_slice(x, lower, upper, N);
    return rows_of(x, lower - 1 + first, n_rows);

}

template <typename T>
inline std::vector<T> array_rows_of_slice(const std::vector<T>& x,
                                          const int lower,
                                          const int upper,
                                          const int N,
                                          const int first,
                                          const int n_rows) {

    check_the_slice(x, lower, upper, N);
    return array_rows_of(x, lower - 1 + first, n_rows);

}

inline std::vector<int> indices_of_chunk_rows(const int n_rows) {

    std::vector<int> index(n_rows);
    for (int i = 0; i < n_rows; ++i) index[i] = i + 1;
    return index;

}

template <typename T_matrix, typename T_vector>
inline const T_matrix& with_matching_group_rows(const T_matrix& matrix,
                                                const T_vector& vector) {

    stan::math::check_size_match("grouped row sum", "coefficient rows", matrix.rows(),
                                 "intercept count", vector.size());
    return matrix;

}

//// ---- a parameter vector given to a product with the rows: struct-of-arrays when it holds vars, else as it is

template <typename T>
inline auto as_struct_of_arrays(const T& x) {

    if constexpr (stan::is_eigen<T>::value && stan::is_var<stan::value_type_t<T>>::value) {
          return stan::math::to_var_value(x);
    } else {
          return x;
    }

}

//// ---- the rows of a data vector, as a struct-of-arrays var vector when the parameters it meets are vars ------

template <typename T_data, typename... T_parameters>
inline auto data_as_struct_of_arrays_if_var(const T_data& x,
                                            const T_parameters&... /*parameters*/) {

    if constexpr (stan::is_var<stan::return_type_t<T_parameters...>>::value) {
          return vector_soa(x);
    } else {
          return x;
    }

}

//// ---- a parameter vector read at an index of the rows (Stan's z[group], 1-based) -----------------------------

//// (each tile gathers only its own rows' values, never the whole vector, and adds their adjoints back at those
////  indices; the tile's values as Stan's struct-of-arrays var vector inside a sum, for the nico_ elementwise
////  overloads, and as an Eigen vector of vars inside a density, which every density takes)
template <typename T_vector, bool struct_of_arrays>
struct gathered_at_rows {
    const T_vector& vector;
    const std::vector<int>& index;
};

template <typename T_vector>
inline gathered_at_rows<T_vector, true> gathered_at_rows_of(const T_vector& vector,
                                                            const std::vector<int>& index) {

    return gathered_at_rows<T_vector, true>{vector, index};

}

template <typename T_vector>
inline gathered_at_rows<T_vector, false> gathered_at_rows_as_vars_of(const T_vector& vector,
                                                                     const std::vector<int>& index) {

    return gathered_at_rows<T_vector, false>{vector, index};

}

template <typename T>
struct is_gathered_at_rows : std::false_type {};

template <typename T_vector, bool struct_of_arrays>
struct is_gathered_at_rows<gathered_at_rows<T_vector, struct_of_arrays>> : std::true_type {};

template <typename T>
struct is_struct_of_arrays_gather : std::false_type {};

template <typename T_vector>
struct is_struct_of_arrays_gather<gathered_at_rows<T_vector, true>> : std::true_type {};

template <typename T>
struct is_matrix_gather : std::false_type {};

template <typename T_vector, bool struct_of_arrays>
struct is_matrix_gather<gathered_at_rows<T_vector, struct_of_arrays>>
      : std::bool_constant<!std::decay_t<T_vector>::IsVectorAtCompileTime> {};

//// ---- a vector of vars read by rows (a parameter vector, or a local made before the loop of the sum) --------

//// (each tile is given a copy of its own rows only, never the whole vector, and adds their adjoints back at
////  those rows, as reduce_sum's sliced argument; the tile's rows as Stan's struct-of-arrays var vector inside a
////  sum, and as an Eigen vector of vars inside a density)
template <typename T_vector, bool struct_of_arrays>
struct sliced_at_rows {
    const T_vector& vector;
};

template <typename T_vector>
inline sliced_at_rows<T_vector, true> sliced_at_rows_of(const T_vector& vector) {

    return sliced_at_rows<T_vector, true>{vector};

}

template <typename T_vector>
inline sliced_at_rows<T_vector, false> sliced_at_rows_as_vars_of(const T_vector& vector) {

    return sliced_at_rows<T_vector, false>{vector};

}

template <typename T>
struct is_sliced_at_rows : std::false_type {};

template <typename T_vector, bool struct_of_arrays>
struct is_sliced_at_rows<sliced_at_rows<T_vector, struct_of_arrays>> : std::true_type {};

template <typename T>
struct is_struct_of_arrays_slice : std::false_type {};

template <typename T_vector>
struct is_struct_of_arrays_slice<sliced_at_rows<T_vector, true>> : std::true_type {};

//// (element i, 0-based, of an Eigen vector or a std::vector)
template <typename T>
inline decltype(auto) element_of(const T& x,
                                 const Eigen::Index i) {

    if constexpr (stan::is_eigen<T>::value) {
          return x.coeff(i);
    } else {
          return x[static_cast<std::size_t>(i)];
    }

}

//// (the vars a shared argument holds: a gathered or sliced vector's are the whole vector's)
template <typename T>
inline const auto& vars_of_shared(const T& shared) {

    if constexpr (is_gathered_at_rows<T>::value || is_sliced_at_rows<T>::value) {
          return shared.vector;
    } else {
          return shared;
    }

}

template <typename T>
inline auto scalar_type_of_shared(const T& shared) {

    return stan::return_type_t<std::decay_t<decltype(vars_of_shared(shared))>>();

}

//// (what the tile is given: a gathered vector's values at the tile's rows, as vars when they are vars; any other
////  shared argument as a copy of its vars on the tile's nested tape, or as it is when it holds no var)
template <typename T>
inline auto shared_for_the_tile(const T& shared,
                                const int first,
                                const int n_rows) {

    if constexpr (is_sliced_at_rows<T>::value) {
          const auto values = Eigen::VectorXd::NullaryExpr(n_rows, [&](const Eigen::Index i) {
                return stan::math::value_of(element_of(shared.vector, first + i));
          });
          if constexpr (!stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
                return Eigen::VectorXd(values);
          } else if constexpr (is_struct_of_arrays_slice<T>::value) {
                return vector_soa(values);
          } else {
                return Eigen::Matrix<stan::math::var, -1, 1>(values.template cast<stan::math::var>());
          }
    } else if constexpr (is_matrix_gather<T>::value) {
          static_assert(!is_struct_of_arrays_gather<T>::value, "matrix gathers use arrays of vars");
          const Eigen::Map<const Eigen::VectorXi> index(shared.index.data() + first, n_rows);
          const int n_groups = static_cast<int>(shared.vector.rows());
          if ((n_rows > 0) && ((index.minCoeff() < 1) || (index.maxCoeff() > n_groups))) {
                const int outside = index.minCoeff() < 1 ? index.minCoeff() : index.maxCoeff();
                stan::math::check_range("matrix[multi] indexing", "index", n_groups, outside);
          }
          const Eigen::MatrixXd values = Eigen::MatrixXd::NullaryExpr(
                n_rows, shared.vector.cols(), [&](const Eigen::Index i, const Eigen::Index j) {
                      return stan::math::value_of(shared.vector.coeff(shared.index[first + i] - 1, j));
                });
          if constexpr (stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
                return Eigen::Matrix<stan::math::var, -1, -1>(values.template cast<stan::math::var>());
          } else {
                return values;
          }
    } else if constexpr (is_gathered_at_rows<T>::value) {
          //// (the tile's indices checked at once, as Stan's z[group] checks them: std::out_of_range if one is
          ////  outside 1, ..., the size of the vector)
          const Eigen::Map<const Eigen::VectorXi> index_of_the_tile(shared.index.data() + first, n_rows);
          const int size_of_the_vector = static_cast<int>(shared.vector.size());
          if ((n_rows > 0) && ((index_of_the_tile.minCoeff() < 1) ||
                               (index_of_the_tile.maxCoeff() > size_of_the_vector))) {
                const int outside = (index_of_the_tile.minCoeff() < 1) ? index_of_the_tile.minCoeff()
                                                                        : index_of_the_tile.maxCoeff();
                stan::math::check_range("vector[multi] indexing", "index", size_of_the_vector, outside);
          }
          const auto values = Eigen::VectorXd::NullaryExpr(n_rows, [&](const Eigen::Index i) {
                return stan::math::value_of(shared.vector.coeff(shared.index[first + i] - 1));
          });
          if constexpr (!stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
                return Eigen::VectorXd(values);
          } else if constexpr (is_struct_of_arrays_gather<T>::value) {
                return vector_soa(values);
          } else {
                return Eigen::Matrix<stan::math::var, -1, 1>(values.template cast<stan::math::var>());
          }
    } else {
          return stan::math::deep_copy_vars(shared);
    }

}

//// (what one chunk of all N rows is given, i.e. no chunking: everything on the main tape, as stanc wrote it,
////  with no nested tape and no copy of a var; a gathered vector's values at the rows as the vector's own vars,
////  or, as a struct-of-arrays var vector, one node whose backward pass adds its adjoints at the indices, 4096 at
////  a time)
template <typename T>
inline auto shared_on_the_main_tape(const T& shared,
                                    const int N) {

    if constexpr (is_sliced_at_rows<T>::value) {
          //// (all N rows: the vector's own vars, or, as a struct-of-arrays var vector, one node whose backward
          ////  pass adds its adjoints to them, 4096 at a time)
          if constexpr (!stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
                return Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(N, [&](const Eigen::Index i) {
                      return stan::math::value_of(element_of(shared.vector, i));
                }));
          } else if constexpr (!is_struct_of_arrays_slice<T>::value) {
                using vector_of_vars = Eigen::Matrix<stan::math::var, -1, 1>;
                return vector_of_vars(vector_of_vars::NullaryExpr(N, [&](const Eigen::Index i) {
                      return element_of(shared.vector, i);
                }));
          } else {
                auto& arena = stan::math::ChainableStack::instance_->memalloc_;
                stan::math::vari** varis = arena.alloc_array<stan::math::vari*>(N);
                for (int first = 0; first < N; first += 4096) {
                      const int last = std::min(first + 4096, N);
                      for (int i = first; i < last; ++i) varis[i] = element_of(shared.vector, i).vi_;
                }
                const vector_soa sliced(Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(N,
                      [&](const Eigen::Index i) { return varis[i]->val_; })));
                stan::math::reverse_pass_callback([sliced, varis, N]() {

                      for (int first = 0; first < N; first += 4096) {
                            const int last = std::min(first + 4096, N);
                            for (int i = first; i < last; ++i) varis[i]->adj_ += sliced.adj().coeff(i);
                      }

                });
                return sliced;
          }
    } else if constexpr (is_matrix_gather<T>::value) {
          static_assert(!is_struct_of_arrays_gather<T>::value, "matrix gathers use arrays of vars");
          const Eigen::Map<const Eigen::VectorXi> index(shared.index.data(), N);
          const int n_groups = static_cast<int>(shared.vector.rows());
          if ((N > 0) && ((index.minCoeff() < 1) || (index.maxCoeff() > n_groups))) {
                const int outside = index.minCoeff() < 1 ? index.minCoeff() : index.maxCoeff();
                stan::math::check_range("matrix[multi] indexing", "index", n_groups, outside);
          }
          using scalar = stan::scalar_type_t<std::decay_t<decltype(shared.vector)>>;
          using matrix = Eigen::Matrix<scalar, -1, -1>;
          return matrix(matrix::NullaryExpr(N, shared.vector.cols(),
                [&](const Eigen::Index i, const Eigen::Index j) {
                      return shared.vector.coeff(shared.index[i] - 1, j);
                }));
    } else if constexpr (is_gathered_at_rows<T>::value) {
          const Eigen::Map<const Eigen::VectorXi> index(shared.index.data(), N);
          const int size_of_the_vector = static_cast<int>(shared.vector.size());
          if ((N > 0) && ((index.minCoeff() < 1) || (index.maxCoeff() > size_of_the_vector))) {
                const int outside = (index.minCoeff() < 1) ? index.minCoeff() : index.maxCoeff();
                stan::math::check_range("vector[multi] indexing", "index", size_of_the_vector, outside);
          }
          if constexpr (!stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
                return Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(N, [&](const Eigen::Index i) {
                      return stan::math::value_of(shared.vector.coeff(shared.index[i] - 1));
                }));
          } else if constexpr (!is_struct_of_arrays_gather<T>::value) {
                using vector_of_vars = Eigen::Matrix<stan::math::var, -1, 1>;
                return vector_of_vars(vector_of_vars::NullaryExpr(N, [&](const Eigen::Index i) {
                      return shared.vector.coeff(shared.index[i] - 1);
                }));
          } else {
                auto& arena = stan::math::ChainableStack::instance_->memalloc_;
                stan::math::vari** varis = arena.alloc_array<stan::math::vari*>(N);
                for (int first = 0; first < N; first += 4096) {
                      const int last = std::min(first + 4096, N);
                      for (int i = first; i < last; ++i) varis[i] = shared.vector.coeff(shared.index[i] - 1).vi_;
                }
                const vector_soa gathered(Eigen::VectorXd(Eigen::VectorXd::NullaryExpr(N,
                      [&](const Eigen::Index i) { return varis[i]->val_; })));
                stan::math::reverse_pass_callback([gathered, varis, N]() {

                      for (int first = 0; first < N; first += 4096) {
                            const int last = std::min(first + 4096, N);
                            for (int i = first; i < last; ++i) varis[i]->adj_ += gathered.adj().coeff(i);
                      }

                });
                return gathered;
          }
    } else {
          return shared;
    }

}

template <typename T, typename T_of_the_tile>
inline void add_adjoints_of_the_tile(double* partials_of_this_shared,
                                     const T& shared,
                                     const T_of_the_tile& shared_of_the_tile,
                                     const int first,
                                     const int n_rows) {

    if constexpr (!stan::is_var<decltype(scalar_type_of_shared(shared))>::value) {
          return;
    } else if constexpr (is_struct_of_arrays_slice<T>::value) {
          const double* adjoints = shared_of_the_tile.vi_->adj_.data();
          for (int i = 0; i < n_rows; ++i) partials_of_this_shared[first + i] += adjoints[i];
    } else if constexpr (is_sliced_at_rows<T>::value) {
          const Eigen::VectorXd adjoints = shared_of_the_tile.adj();
          for (int i = 0; i < n_rows; ++i) partials_of_this_shared[first + i] += adjoints[i];
    } else if constexpr (is_matrix_gather<T>::value) {
          const Eigen::MatrixXd adjoints = shared_of_the_tile.adj();
          for (Eigen::Index j = 0; j < shared.vector.cols(); ++j) {
                for (int i = 0; i < n_rows; ++i) {
                      const Eigen::Index offset = shared.index[first + i] - 1 + j * shared.vector.rows();
                      partials_of_this_shared[offset] += adjoints.coeff(i, j);
                }
          }
    } else if constexpr (is_struct_of_arrays_gather<T>::value) {
          const double* adjoints = shared_of_the_tile.vi_->adj_.data();
          for (int i = 0; i < n_rows; ++i) partials_of_this_shared[shared.index[first + i] - 1] += adjoints[i];
    } else if constexpr (is_gathered_at_rows<T>::value) {
          const Eigen::VectorXd adjoints = shared_of_the_tile.adj();
          for (int i = 0; i < n_rows; ++i) partials_of_this_shared[shared.index[first + i] - 1] += adjoints[i];
    } else {
          stan::math::accumulate_adjoints(partials_of_this_shared, shared_of_the_tile);
    }

}

template <std::size_t... I, typename Of_the_tile, typename... Shared>
inline void add_adjoints_of_every_shared(std::index_sequence<I...> /*indices*/,
                                         double* partials,
                                         const std::array<std::size_t, sizeof...(Shared)>& offsets,
                                         const Of_the_tile& shared_of_the_tile,
                                         const int first,
                                         const int n_rows,
                                         const Shared&... shared) {

    (add_adjoints_of_the_tile(partials + offsets[I], shared, std::get<I>(shared_of_the_tile), first, n_rows),
     ...);

}

//// ---- the vari of each shared var, saved chunk by chunk, 4096 at a time --------------------------------------

template <typename T>
inline stan::math::vari** save_varis_chunk_by_chunk(stan::math::vari** varis,
                                                     const T& x) {

    using T_plain = std::decay_t<T>;
    if constexpr (std::is_same<T_plain, stan::math::var>::value) {
          *varis = x.vi_;
          return varis + 1;
    } else if constexpr ((stan::is_eigen<T_plain>::value || stan::is_std_vector<T_plain>::value) &&
                         std::is_same<stan::value_type_t<T_plain>, stan::math::var>::value) {
          const std::size_t n = static_cast<std::size_t>(x.size());
          for (std::size_t first = 0; first < n; first += 4096) {
                const std::size_t last = std::min<std::size_t>(first + 4096, n);
                for (std::size_t i = first; i < last; ++i) {
                      if constexpr (stan::is_eigen<T_plain>::value) {
                            varis[i] = x.coeff(static_cast<Eigen::Index>(i)).vi_;
                      } else {
                            varis[i] = x[i].vi_;
                      }
                }
          }
          return varis + n;
    } else {
          //// (no var, or a shape Stan's own save_varis walks: as Stan does it)
          return stan::math::save_varis(varis, x);
    }

}

//// ---- the chunks of rows 0 .. N - 1 that reduce_sum_static makes with this chunk size -----------------------
//// (TBB's simple partitioner: a piece of more than chunk_size rows is halved at first + size / 2, the left half
////  first; the chunks, left to right)

struct chunk_of_rows {

    int first;
    int n_rows;

};

inline void add_the_chunks_of_reduce_sum_static(const int first,
                                                const int n_rows,
                                                const int chunk_size,
                                                std::vector<chunk_of_rows>& chunks) {

    if (n_rows <= chunk_size) {
          chunks.push_back({first, n_rows});
          return;
    }
    const int n_rows_of_the_left_half = n_rows / 2;
    add_the_chunks_of_reduce_sum_static(first, n_rows_of_the_left_half, chunk_size, chunks);
    add_the_chunks_of_reduce_sum_static(first + n_rows_of_the_left_half, n_rows - n_rows_of_the_left_half,
                                        chunk_size, chunks);

}

inline std::vector<chunk_of_rows> chunks_of_reduce_sum_static(const int N,
                                                              const int chunk_size) {

    if (chunk_size < 1) {
          throw std::domain_error("chunk_size_of_auto_chunking must be at least 1");
    }
    std::vector<chunk_of_rows> chunks;
    if (N > 0) add_the_chunks_of_reduce_sum_static(0, N, chunk_size, chunks);
    return chunks;

}

//// (chunk_size_of_auto_chunking = ceiling(N / N_chunks_of_auto_chunking), at least 1)
inline int chunk_size_of_auto_chunking(const int N,
                                       const int N_chunks_of_auto_chunking) {

    if (N_chunks_of_auto_chunking < 1) {
          throw std::domain_error("N_chunks_of_auto_chunking must be at least 1");
    }
    return std::max(1, (N + N_chunks_of_auto_chunking - 1) / N_chunks_of_auto_chunking);

}

//// (N_chunks_of_auto_chunking from the model's data, read once when the model is made; a template, so that the
////  data's context type is known only where the model's constructor calls it)
template <typename Context>
inline int N_chunks_of_auto_chunking_from_the_data(const Context& context) {

    if (!context.contains_i("N_chunks_of_auto_chunking") ||
        context.vals_i("N_chunks_of_auto_chunking").empty()) {
          throw std::domain_error("the data must hold the integer N_chunks_of_auto_chunking");
    }
    const int N_chunks_of_auto_chunking = context.vals_i("N_chunks_of_auto_chunking")[0];
    if (N_chunks_of_auto_chunking < 1) {
          throw std::domain_error("N_chunks_of_auto_chunking must be at least 1");
    }
    return N_chunks_of_auto_chunking;

}

//// ---- the sum over rows 0 .. N - 1, chunk by chunk ----------------------------------------------------------

template <typename F, typename... Shared>
inline auto sum_of_rows_chunk_by_chunk(const std::vector<chunk_of_rows>& chunks,
                                       const F& sum_of_rows,
                                       const Shared&... shared) {

    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    //// (with no var among the shared arguments, as in log_prob<double>: the plain sum, chunk by chunk)
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {

          double value = 0.0;
          for (const chunk_of_rows& chunk : chunks) {
                const int first = chunk.first;
                const int n_rows = chunk.n_rows;
                value += stan::math::apply([&](auto&&... of_the_tile) { return sum_of_rows(first, n_rows,
                                                                                            of_the_tile...); },
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

                const int first = chunk.first;
                const int n_rows = chunk.n_rows;
                const stan::math::nested_rev_autodiff tile;
                auto shared_of_the_tile = std::make_tuple(shared_for_the_tile(shared, first, n_rows)...);
                stan::math::var sum_of_tile = stan::math::apply(
                      [&](auto&&... of_the_tile) { return sum_of_rows(first, n_rows, of_the_tile...); },
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

//// ---- a sliced vector has exactly the N rows of the sum -----------------------------------------------------

template <typename T>
inline void check_the_rows_of_shared(const T& shared,
                                     const int N) {

    if constexpr (is_sliced_at_rows<T>::value) {
          stan::math::check_size_match("auto_chunking", "rows of the sum", N, "size of the sliced vector",
                                       static_cast<int>(shared.vector.size()));
    }

}

//// ---- the sum over rows 0 .. N - 1 in the chunks reduce_sum_static makes with chunk_size_of_auto_chunking --

template <typename F, typename... Shared>
inline auto sum_of_rows_in_the_chunks_of_reduce_sum_static(const int N,
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
    return sum_of_rows_chunk_by_chunk(chunks, sum_of_rows, shared...);

}

//// ---- the same, with stanc's own statement given (a statement whose N-row locals stay where stanc made them):
////      one chunk of all the rows runs that statement exactly as stanc wrote it, so that AUTO-CHUNKING at one
////      chunk is the model itself (no view of the rows, no copy of a shared vector)

template <typename G, typename F, typename... Shared>
inline auto sum_of_rows_in_the_chunks_of_reduce_sum_static_or_as_stanc_wrote_it(
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
    return sum_of_rows_in_the_chunks_of_reduce_sum_static(N, N_chunks_of_auto_chunking, sum_of_rows, shared...);

}

//// ---- one elementwise nico_ function of a struct-of-arrays var vector ----------------------------------------

enum class values_read_by_the_derivative { operand, result, both };

template <values_read_by_the_derivative needed, typename K, typename D>
inline vector_soa elementwise_on_soa(const vector_soa& x,
                                     K in_place_kernel,
                                     D derivative_function) {

    const int n = static_cast<int>(x.size());
    //// (from a Map, so the result's values are a new copy: an arena_matrix argument would share x's values, and
    ////  the kernel would overwrite them)
    vector_soa result(Eigen::Map<const Eigen::VectorXd>(x.vi_->val_.data(), n));
    in_place_kernel(result.vi_->val_.data(), n);
    stan::math::reverse_pass_callback([x, result, n, derivative_function]() mutable {

          alignas(64) double derivatives[64];
          const double* values_of_x = x.vi_->val_.data();
          const double* values_of_result = result.vi_->val_.data();
          double* adjoints_of_x = x.vi_->adj_.data();
          const double* adjoints_of_result = result.vi_->adj_.data();
          for (int first = 0; first < n; first += 64) {

                const int count = std::min(64, n - first);
                const bool reads_x = (needed != values_read_by_the_derivative::result);
                const bool reads_result = (needed != values_read_by_the_derivative::operand);
                derivative_function(reads_x ? values_of_x + first : nullptr,
                                    reads_result ? values_of_result + first : nullptr,
                                    count, derivatives);
                for (int i = 0; i < count; ++i) {
                      adjoints_of_x[first + i] += adjoints_of_result[first + i] * derivatives[i];
                }

          }

    });
    return result;

}

}  // namespace nico::math::auto_chunking

//// ---- the nico_ elementwise functions of a struct-of-arrays var vector, under their Stan-facing names --------

#define NICO_AUTO_CHUNKING_ELEMENTWISE(NAME, KERNEL, DERIVATIVE, NEEDED)                                       \
namespace nico::math {                                                                                         \
inline auto_chunking::vector_soa NAME(const auto_chunking::vector_soa& x,                                      \
                                      std::ostream* /*pstream__*/) {                                           \
                                                                                                               \
    return auto_chunking::elementwise_on_soa<auto_chunking::values_read_by_the_derivative::NEEDED>(            \
          x, [](double* values, int n) { nico::math::KERNEL(values, n); },                                     \
          [](const double* a, const double* f, int n, double* d) { nico::math::DERIVATIVE(a, f, n, d); });     \
                                                                                                               \
}                                                                                                              \
}                                                                                                              \
using nico::math::NAME;

NICO_AUTO_CHUNKING_ELEMENTWISE(nico_exp,           k_exp,           d_exp,           result)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_log,           k_log,           d_log,           operand)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_log1p,         k_log1p,         d_log1p,         operand)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_log1m,         k_log1m,         d_log1m,         operand)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_log1p_exp,     k_log1p_exp,     d_log1p_exp,     operand)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_inv_logit,     k_inv_logit,     d_inv_logit,     result)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_log_inv_logit, k_log_inv_logit, d_log_inv_logit, operand)
NICO_AUTO_CHUNKING_ELEMENTWISE(nico_Phi,           k_Phi,           d_Phi,           operand)

#undef NICO_AUTO_CHUNKING_ELEMENTWISE
