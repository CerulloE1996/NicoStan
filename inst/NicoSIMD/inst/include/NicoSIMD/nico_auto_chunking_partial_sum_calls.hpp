#pragma once

//// ---- AUTO-CHUNKING of a direct call of a reduce_sum partial-sum function on all its rows ---------------------
////
//// A partial-sum function f(x_slice, start, end, ...) given to reduce_sum / reduce_sum_static is, by its contract,
//// the sum over the rows start .. end of its sliced argument x; a direct call f(x, 1, size(x), ...) (the model's
//// branch with one chunk) is therefore the sum of f over any split of the rows. Each tile is given a copy of its
//// own rows of x only, and adds their adjoints back at those rows, as reduce_sum's sliced argument; the chunks are
//// reduce_sum_static's (nico_auto_chunking.hpp).

namespace nico::math::auto_chunking {

//// ---- the sliced argument of a partial-sum function: an array of rows, and the first var of each row ---------

template <typename T_array>
struct sliced_array_at_rows {

    const T_array& array;
    std::vector<std::size_t> first_var_of_row;

};

template <typename T_array>
inline sliced_array_at_rows<T_array> sliced_array_at_rows_of(const T_array& array) {

    const std::size_t n = array.size();
    std::vector<std::size_t> first_var_of_row(n + 1, 0);
    //// (the rows' var counts added up chunk by chunk, 4096 rows at a time)
    for (std::size_t first = 0; first < n; first += 4096) {
          const std::size_t last = std::min<std::size_t>(first + 4096, n);
          for (std::size_t i = first; i < last; ++i) {
                first_var_of_row[i + 1] = first_var_of_row[i] + stan::math::count_vars(array[i]);
          }
    }
    return sliced_array_at_rows<T_array>{array, std::move(first_var_of_row)};

}

//// (the vars it holds: the whole array's, in the order Stan's count_vars and save_varis walk them)
template <typename T_array>
inline const T_array& vars_of_shared(const sliced_array_at_rows<T_array>& shared) {

    return shared.array;

}

//// (what the tile is given: a copy of its own rows, as new vars on the tile's nested tape when they are vars)
template <typename T_array>
inline auto shared_for_the_tile(const sliced_array_at_rows<T_array>& shared,
                                const int first,
                                const int n_rows) {

    std::decay_t<T_array> rows_of_the_tile(shared.array.begin() + first,
                                           shared.array.begin() + first + n_rows);
    return stan::math::deep_copy_vars(rows_of_the_tile);

}

//// (one chunk of all the rows: the array itself, on the main tape)
template <typename T_array>
inline const T_array& shared_on_the_main_tape(const sliced_array_at_rows<T_array>& shared,
                                              const int /*N*/) {

    return shared.array;

}

template <typename T_array, typename T_of_the_tile>
inline void add_adjoints_of_the_tile(double* partials_of_this_shared,
                                     const sliced_array_at_rows<T_array>& shared,
                                     const T_of_the_tile& rows_of_the_tile,
                                     const int first,
                                     const int /*n_rows*/) {

    if constexpr (stan::is_var<stan::scalar_type_t<std::decay_t<T_array>>>::value) {
          stan::math::accumulate_adjoints(partials_of_this_shared + shared.first_var_of_row[first],
                                          rows_of_the_tile);
    }

}

template <typename T_array>
inline void check_the_rows_of_shared(const sliced_array_at_rows<T_array>& shared,
                                     const int N) {

    stan::math::check_size_match("auto_chunking", "rows of the sum", N, "size of the sliced array",
                                 static_cast<int>(shared.array.size()));

}

}  // namespace nico::math::auto_chunking
