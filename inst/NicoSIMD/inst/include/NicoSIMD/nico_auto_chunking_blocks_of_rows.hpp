#pragma once

//// ---- AUTO-CHUNKING of a block of the model that reads its rows 1 .. size(x) of a row array x -----------------
////
//// A block of statements (stanc's C++ of the model's { ... } block) whose only effect outside it is to
//// add to the target, and whose rows are the rows 1 .. size(x) of a row array x, runs on each chunk of
//// rows of the chunks reduce_sum_static makes (nico_auto_chunking.hpp), as its own statements, unchanged:
//// inside a tile, x is a copy of the tile's own rows, every data object with N entries in a dimension is
//// read at the tile's rows only, in the type stanc gave it (a Map of a vector's entries or of a copy of a
//// matrix's rows, a copy of an array's rows), and every other var the block reads is copied into the tile's
//// nested tape. One chunk runs the block on all its rows on the main tape, on the model's own objects: stanc's
//// statement. The first call with chunks is checked, at the end of log_prob, against stanc's own log_prob of
//// the values (no tape): if they differ (a block that is not a sum over its rows, e.g. one with a risk set over
//// all the rows), every block runs as stanc wrote it from then on, and that call is made again.
//// After nico_auto_chunking.hpp in the model's user header.

#include <stan/math/rev.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <stdexcept>
#include <vector>

namespace nico::math::auto_chunking {

//// ---- the data of a block at the rows of a tile -----------------------------------------------------------------
//// (each in the type stanc gave it: a function of the model that takes the data as stanc declared it takes them
////  at the tile's rows too, and is the same function stanc's statement calls)

//// (an array of the data: a copy of its tile's rows, in its own type, when it has N entries in a dimension at
////  any depth; else the array itself)
template <typename T>
struct array_of_the_data_at_the_rows_of_a_tile {

    const std::vector<T>* whole;
    std::vector<T> rows;

};

//// (a vector or matrix of the data, which stanc keeps as an Eigen::Map: the tile's is the same Map, of the
////  tile's rows in place (a vector's entries, a matrix's columns), or of a copy of them (a matrix's rows, which
////  are not contiguous in a column-major matrix), else of all of it)
template <typename T>
struct map_of_the_data_at_the_rows_of_a_tile {

    const double* data;
    Eigen::Index rows;
    Eigen::Index columns;
    Eigen::Matrix<double, -1, -1> copy_of_the_rows;

};

template <typename T>
inline const T& the_data(const T& x) {

    return x;

}

template <typename T>
inline const std::vector<T>& the_data(const array_of_the_data_at_the_rows_of_a_tile<T>& x) {

    return (x.whole != nullptr) ? *x.whole : x.rows;

}

template <typename T>
inline T the_data(const map_of_the_data_at_the_rows_of_a_tile<T>& x) {

    double* data = const_cast<double*>((x.data != nullptr) ? x.data : x.copy_of_the_rows.data());
    if constexpr ((T::ColsAtCompileTime == 1) || (T::RowsAtCompileTime == 1)) {
          return T(data, x.rows * x.columns);
    } else {
          return T(data, x.rows, x.columns);
    }

}

//// (whether an array of the data, at any depth, has N entries in a dimension: its own, or a vector's entries, a
////  matrix's rows or columns inside it)
template <typename T>
inline bool has_N_rows_at_any_depth(const T& x,
                                    const int N) {

    if constexpr (stan::is_eigen<T>::value) {
          if constexpr ((std::decay_t<T>::ColsAtCompileTime == 1) || (std::decay_t<T>::RowsAtCompileTime == 1)) {
                return static_cast<int>(x.size()) == N;
          } else {
                return (static_cast<int>(x.rows()) == N) || (static_cast<int>(x.cols()) == N);
          }
    } else if constexpr (stan::is_std_vector<T>::value) {
          if (static_cast<int>(x.size()) == N) return true;
          return std::any_of(x.begin(), x.end(), [N](const auto& element) {
                return has_N_rows_at_any_depth(element, N);
          });
    } else {
          return false;
    }

}

//// (the same data at the tile's rows, in its own type, at any depth: an array with N entries, its rows; a vector
////  with N entries, its rows; a matrix with N rows (else N columns), its rows (columns); everything else as
////  it is)
template <typename T>
inline T rows_of_the_data_in_its_own_type(const T& x,
                                          const int N,
                                          const int first,
                                          const int n_rows) {

    if constexpr (stan::is_eigen<T>::value) {
          if constexpr ((std::decay_t<T>::ColsAtCompileTime == 1) || (std::decay_t<T>::RowsAtCompileTime == 1)) {
                return (static_cast<int>(x.size()) == N) ? T(x.segment(first, n_rows)) : x;
          } else {
                if (static_cast<int>(x.rows()) == N) return x.middleRows(first, n_rows);
                if (static_cast<int>(x.cols()) == N) return x.middleCols(first, n_rows);
                return x;
          }
    } else if constexpr (stan::is_std_vector<T>::value) {
          if (static_cast<int>(x.size()) == N) return T(x.begin() + first, x.begin() + first + n_rows);
          T rows;
          rows.reserve(x.size());
          for (const auto& element : x) {
                rows.push_back(rows_of_the_data_in_its_own_type(element, N, first, n_rows));
          }
          return rows;
    } else {
          return x;
    }

}

template <typename T, stan::require_eigen_t<T>* = nullptr>
inline auto data_at_the_rows_of_the_tile(const T& x,
                                         const int N,
                                         const int first,
                                         const int n_rows) {

    using T_plain = std::decay_t<T>;
    constexpr bool is_a_vector = (T_plain::ColsAtCompileTime == 1) || (T_plain::RowsAtCompileTime == 1);
    constexpr bool is_a_map = is_a_vector ? std::is_constructible<T_plain, double*, Eigen::Index>::value
                                          : std::is_constructible<T_plain, double*, Eigen::Index,
                                                                  Eigen::Index>::value;
    if constexpr (!is_a_map) {
          //// (a vector or matrix of its own: a copy of the tile's rows, in its own type)
          return rows_of_the_data_in_its_own_type(T_plain(x), N, first, n_rows);
    } else {
          static_assert(!T_plain::IsRowMajor, "the data's matrices are column-major");
          map_of_the_data_at_the_rows_of_a_tile<T_plain> of_the_tile{x.data(), x.rows(), x.cols(), {}};
          //// (all the rows: the data itself)
          if ((first == 0) && (n_rows == N)) return of_the_tile;
          if constexpr (is_a_vector) {
                if (static_cast<int>(x.size()) == N) {
                      of_the_tile.data = x.data() + first;
                      of_the_tile.rows = (T_plain::ColsAtCompileTime == 1) ? n_rows : 1;
                      of_the_tile.columns = (T_plain::ColsAtCompileTime == 1) ? 1 : n_rows;
                }
          } else {
                if (x.outerStride() != x.rows()) {
                      throw std::domain_error("auto_chunking: a matrix of the data that is not contiguous");
                }
                if (static_cast<int>(x.rows()) == N) {
                      of_the_tile.data = nullptr;
                      of_the_tile.copy_of_the_rows = x.middleRows(first, n_rows);
                      of_the_tile.rows = n_rows;
                } else if (static_cast<int>(x.cols()) == N) {
                      of_the_tile.data = x.data() + static_cast<Eigen::Index>(first) * x.rows();
                      of_the_tile.columns = n_rows;
                }
          }
          return of_the_tile;
    }

}

template <typename T>
inline auto data_at_the_rows_of_the_tile(const std::vector<T>& x,
                                         const int N,
                                         const int first,
                                         const int n_rows) {

    array_of_the_data_at_the_rows_of_a_tile<T> of_the_tile{&x, {}};
    //// (all the rows: the array itself)
    if ((first == 0) && (n_rows == N)) return of_the_tile;
    if (has_N_rows_at_any_depth(x, N)) {
          of_the_tile.whole = nullptr;
          of_the_tile.rows = rows_of_the_data_in_its_own_type(x, N, first, n_rows);
    }
    return of_the_tile;

}

//// (an array or vector of the data read in blocks of N rows, the rows of one group (an item, a class) after another:
////  in a tile, the tile's rows of each block, one block after another, so that the block's own indices, with N the
////  tile's number of rows, read them; at all the rows, the data itself)
template <typename T>
inline auto data_in_blocks_at_the_rows_of_the_tile(const std::vector<T>& x,
                                                   const int N,
                                                   const int first,
                                                   const int n_rows) {

    array_of_the_data_at_the_rows_of_a_tile<T> of_the_tile{&x, {}};
    if ((first == 0) && (n_rows == N)) return of_the_tile;
    if ((N <= 0) || ((static_cast<int>(x.size()) % N) != 0)) {
          throw std::domain_error("auto_chunking: data read in blocks of N rows whose size is not a multiple of N");
    }
    const std::size_t n_blocks = x.size() / static_cast<std::size_t>(N);
    of_the_tile.whole = nullptr;
    of_the_tile.rows.reserve(n_blocks * static_cast<std::size_t>(n_rows));
    for (std::size_t block = 0; block < n_blocks; ++block) {
          const auto start = x.begin() + block * static_cast<std::size_t>(N) + first;
          of_the_tile.rows.insert(of_the_tile.rows.end(), start, start + n_rows);
    }
    return of_the_tile;

}

template <typename T, stan::require_eigen_vector_t<T>* = nullptr>
inline auto data_in_blocks_at_the_rows_of_the_tile(const T& x,
                                                   const int N,
                                                   const int first,
                                                   const int n_rows) {

    using T_plain = std::decay_t<T>;
    map_of_the_data_at_the_rows_of_a_tile<T_plain> of_the_tile{x.data(), x.rows(), x.cols(), {}};
    if ((first == 0) && (n_rows == N)) return of_the_tile;
    if ((N <= 0) || ((static_cast<int>(x.size()) % N) != 0)) {
          throw std::domain_error("auto_chunking: data read in blocks of N rows whose size is not a multiple of N");
    }
    const Eigen::Index n_blocks = x.size() / N;
    of_the_tile.data = nullptr;
    of_the_tile.copy_of_the_rows.resize(n_blocks * n_rows, 1);
    for (Eigen::Index block = 0; block < n_blocks; ++block) {
          of_the_tile.copy_of_the_rows.middleRows(block * n_rows, n_rows)
                = x.segment(block * N + first, n_rows);
    }
    of_the_tile.rows = (T_plain::ColsAtCompileTime == 1) ? n_blocks * n_rows : 1;
    of_the_tile.columns = (T_plain::ColsAtCompileTime == 1) ? 1 : n_blocks * n_rows;
    return of_the_tile;

}

//// ---- the row array x of a block: each tile is given a copy of its own rows only, and adds their adjoints back at
////      those rows (a Stan array is rectangular: every row holds as many vars as the first, so the tile's first
////      var is first * that number, with nothing of size N kept) ---------------------------------------------------

template <typename T_array>
struct rows_of_the_array_of_a_block {

    const T_array& array;

};

template <typename T_array>
inline rows_of_the_array_of_a_block<T_array> rows_of_the_array_of_a_block_of(const T_array& array) {

    return rows_of_the_array_of_a_block<T_array>{array};

}

template <typename T>
struct is_rows_of_the_array_of_a_block : std::false_type {};

template <typename T_array>
struct is_rows_of_the_array_of_a_block<rows_of_the_array_of_a_block<T_array>> : std::true_type {};

template <typename T_array>
inline const T_array& vars_of_shared(const rows_of_the_array_of_a_block<T_array>& shared) {

    return shared.array;

}

template <typename T_array>
inline auto shared_for_the_tile(const rows_of_the_array_of_a_block<T_array>& shared,
                                const int first,
                                const int n_rows) {

    std::decay_t<T_array> rows_of_the_tile(shared.array.begin() + first, shared.array.begin() + first + n_rows);
    return stan::math::deep_copy_vars(rows_of_the_tile);

}

template <typename T_array>
inline const T_array& shared_on_the_main_tape(const rows_of_the_array_of_a_block<T_array>& shared,
                                              const int /*N*/) {

    return shared.array;

}

template <typename T_array, typename T_of_the_tile>
inline void add_adjoints_of_the_tile(double* partials_of_this_shared,
                                     const rows_of_the_array_of_a_block<T_array>& shared,
                                     const T_of_the_tile& rows_of_the_tile,
                                     const int first,
                                     const int /*n_rows*/) {

    if constexpr (stan::is_var<stan::scalar_type_t<std::decay_t<T_array>>>::value) {
          const std::size_t vars_per_row = shared.array.empty() ? 0 : stan::math::count_vars(shared.array[0]);
          const std::size_t first_var = static_cast<std::size_t>(first) * vars_per_row;
          stan::math::accumulate_adjoints(partials_of_this_shared + first_var,
                                          rows_of_the_tile);
    }

}

template <typename T_array>
inline void check_the_rows_of_shared(const rows_of_the_array_of_a_block<T_array>& shared,
                                     const int N) {

    stan::math::check_size_match("auto_chunking", "rows of the block", N, "size of the row array",
                                 static_cast<int>(shared.array.size()));

}

//// ---- a var the block reads that is not its rows: one with N entries is never copied whole into a tile ----------

template <typename T>
inline bool has_N_entries_in_its_first_dimension(const T& x,
                                                 const int N) {

    if constexpr (is_rows_of_the_array_of_a_block<T>::value || is_sliced_at_rows<T>::value) {
          return false;
    } else if constexpr (stan::is_std_vector<T>::value) {
          return static_cast<int>(x.size()) == N;
    } else if constexpr (stan::is_eigen<T>::value) {
          return (static_cast<int>(x.rows()) == N) || (static_cast<int>(x.cols()) == N);
    } else {
          return false;
    }

}

//// ---- the model's own object of a shared argument, as the block reads it: on the main tape, no copy ----------

template <typename T>
inline auto& model_object_of_shared(const T& shared) {

    if constexpr (is_rows_of_the_array_of_a_block<T>::value) {
          return const_cast<std::decay_t<decltype(shared.array)>&>(shared.array);
    } else if constexpr (is_sliced_at_rows<T>::value) {
          return const_cast<std::decay_t<decltype(shared.vector)>&>(shared.vector);
    } else {
          return const_cast<T&>(shared);
    }

}

//// ---- the state of a block: 0 not run on chunks yet, 3 run on chunks and not yet checked, 1 checked, 2 run as
////      stanc wrote it -------------------------------------------------------------------------------------------

template <typename... States>
inline bool a_block_of_rows_is_not_yet_checked(const States&... state_of_each_block) {

    return ((state_of_each_block.load(std::memory_order_relaxed) == 3) || ...);

}

//// (while the check evaluates the chunks' log density with every term (propto = false) inside a call with
////  propto = true, that inner call returns its chunks' log density as it is, with no check of its own)
inline thread_local bool the_chunks_of_the_blocks_are_being_checked = false;

struct check_of_the_chunks_of_the_blocks {

    check_of_the_chunks_of_the_blocks() { the_chunks_of_the_blocks_are_being_checked = true; }
    ~check_of_the_chunks_of_the_blocks() { the_chunks_of_the_blocks_are_being_checked = false; }

};

//// (the check's outcome: 2, every block as stanc wrote it; 1, the blocks run on chunks are checked; 3,
////  not finite,
////  checked at a later call)
template <typename... States>
inline void record_the_check_of_the_blocks_of_rows(const int state,
                                                   States&... state_of_each_block) {

    const auto record = [state](std::atomic<int>& state_of_the_block) {
          if (state == 2) {
                state_of_the_block.store(2, std::memory_order_relaxed);
          } else if ((state == 1) && (state_of_the_block.load(std::memory_order_relaxed) == 3)) {
                state_of_the_block.store(1, std::memory_order_relaxed);
          }
    };
    (record(state_of_each_block), ...);

}

//// (the log density of the chunks against stanc's own log_prob of the values; a value that is not finite is
////  checked at a later call)
inline int state_after_the_check(const double log_density_of_the_chunks,
                                 const double log_density_of_stanc) {

    if (!std::isfinite(log_density_of_the_chunks) || !std::isfinite(log_density_of_stanc)) return 3;
    const double scale = std::max({1.0, std::fabs(log_density_of_the_chunks), std::fabs(log_density_of_stanc)});
    if (std::fabs(log_density_of_the_chunks - log_density_of_stanc) <= 1e-8 * scale) return 1;
    std::cerr << "AUTO-CHUNKING: the chunks of the blocks of rows give the log density "
              << log_density_of_the_chunks << ", stanc's own log_prob " << log_density_of_stanc
              << "; the blocks are run as stanc wrote them" << std::endl;
    return 2;

}

//// ---- the block's total over the chunks of reduce_sum_static ------------------------------------------------
//// (one chunk of all the rows, no var, or a block run as stanc wrote it: the block on all its rows on the main
////  tape, on the model's own objects; a var it reads with N entries that are not its rows: the same, from then
////  on, with a message)

template <typename F, typename... Shared>
inline auto sum_of_a_block_of_rows_in_the_chunks_of_reduce_sum_static(const char* name_of_the_block,
                                                                       const int N,
                                                                       const int N_chunks_of_auto_chunking,
                                                                       std::atomic<int>& state_of_the_block,
                                                                       const F& block_on_rows,
                                                                       const Shared&... shared) {

    (check_the_rows_of_shared(shared, N), ...);
    const auto on_all_its_rows_on_the_main_tape = [&]() {
          auto model_objects = std::forward_as_tuple(model_object_of_shared(shared)...);
          return stan::math::apply(
                [&](auto&&... of_the_model) { return block_on_rows(0, N, of_the_model...); }, model_objects);
    };
    using scalar_of_the_sum = stan::return_type_t<decltype(scalar_type_of_shared(shared))...>;
    if constexpr (!stan::is_var<scalar_of_the_sum>::value) {
          return on_all_its_rows_on_the_main_tape();
    } else {
          //// (one call of the block on the main tape and one loop over the chunks, so that the block's code is
          ////  made once)
          bool on_the_main_tape = (chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking) >= N)
                                  || (state_of_the_block.load(std::memory_order_relaxed) == 2);
          if (!on_the_main_tape && (has_N_entries_in_its_first_dimension(shared, N) || ...)) {
                state_of_the_block.store(2, std::memory_order_relaxed);
                std::cerr << "AUTO-CHUNKING: " << name_of_the_block << " reads a var of N rows that is not "
                          << "its rows; the block is run as stanc wrote it" << std::endl;
                on_the_main_tape = true;
          }
          if (!on_the_main_tape) {
                const std::vector<chunk_of_rows> chunks
                      = chunks_of_reduce_sum_static(N, chunk_size_of_auto_chunking(N, N_chunks_of_auto_chunking));
                const bool checked = (state_of_the_block.load(std::memory_order_relaxed) == 1);
                //// (not checked yet: an error on the chunks makes the block run as stanc wrote it)
                try {
                      stan::math::var chunked = sum_of_rows_chunk_by_chunk(chunks, block_on_rows, shared...);
                      if (!checked) state_of_the_block.store(3, std::memory_order_relaxed);
                      return chunked;
                } catch (const std::exception& error) {
                      if (checked) throw;
                      //// (stanc's statement on all the rows: an error of the model itself is thrown again from it,
                      ////  and the block's state is left as it was)
                      stan::math::var of_stanc = stan::math::var(on_all_its_rows_on_the_main_tape());
                      state_of_the_block.store(2, std::memory_order_relaxed);
                      std::cerr << "AUTO-CHUNKING: " << name_of_the_block << " failed on its chunks ("
                                << error.what() << "); the block is run as stanc wrote it" << std::endl;
                      return of_stanc;
                }
          }
          return stan::math::var(on_all_its_rows_on_the_main_tape());
    }

}

}  // namespace nico::math::auto_chunking
