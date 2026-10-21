#pragma once

// The readers of the nuisance block that NicoStan samples for a model in NicoStan's nuisance-block form
// (reduce_sum_chunked: needs the private runtime with options(NicoStan.reduce_sum_nuisance_block = TRUE); see
// reduce_sum_nuisance_block_owned_by_NicoStan.inc). NicoStan's nuisance values are one array: the model's
// nuisance blocks one after the other, each in Stan's own order of its values (an array of vectors or
// row_vectors: term by term, the values_per_term values of each term together). Inside a chunk of a reduce_sum,
// the values become nodes of the chunk's own tape; the adjoints of the values of the chunk's own terms (the
// terms of reduce_sum's sliced argument the chunk was given) go straight into NicoStan's array of the nuisance
// gradient, those of other terms are added after all the chunks. On the main tape (e.g. the few values a model
// reads at the boundaries of its chunks), they become nodes of the main tape, added after the gradient. Outside
// NicoStan's gradient call (e.g. through BridgeStan's own interface), the values set with
// nicostan_set_the_nuisance_values_used_outside_NicoStan_R are used, and their gradient is not kept. "type" only
// carries the scalar type of the caller (any real of the model, e.g. a main parameter): with a plain number the
// values are returned as numbers. Declared in the Stan functions block (no body); terms and values are 1-based,
// offset is the number of values of the blocks before this one:
//     vector nicostan_nuisance_values(int first, int last, real type);
//     vector nicostan_nuisance_values_at(array[] int units, real type);
//         (the values first..last, or the values of the units of an array, of a block of one value per term)
//     real nicostan_nuisance_value(int term, int offset, real type);
//     array[] real nicostan_nuisance_array_of_reals(int first_term, int last_term, int offset, real type);
//     array[] real nicostan_nuisance_array_of_reals_at(array[] int terms, int offset, real type);
//     vector nicostan_nuisance_vector(int first_term, int last_term, int offset, real type);
//     vector nicostan_nuisance_vector_at(array[] int terms, int offset, real type);
//     array[] vector nicostan_nuisance_array_of_vectors(int first_term, int last_term, int values_per_term,
//                                                       int offset, real type);
//     array[] row_vector nicostan_nuisance_array_of_row_vectors(int first_term, int last_term,
//                                                               int values_per_term, int offset, real type);
//     matrix nicostan_nuisance_matrix_of_terms(int first_term, int last_term, int values_per_term, int offset,
//                                              real type);
//         (the terms first_term..last_term as the rows of one matrix, the values_per_term values of each term
//          along its row: the same values as to_matrix() of nicostan_nuisance_array_of_row_vectors(), in one
//          allocation, without a row_vector of its own for every term)
//     array[] matrix nicostan_nuisance_array_of_matrices(int first_term, int last_term, int rows, int columns,
//                                                        int offset, real type);
//     <type> nicostan_constrain(<type> x, int kind, real a, real b);
//     real nicostan_log_jacobian_of_constraint(<type> x, int kind, real a, real b);
//         (a constrained block: NicoStan holds its unconstrained values; see the end of this file)

#include <stan/math/rev.hpp>
#include "reuse_of_nuisance_read_nodes_within_one_map_job.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace nicostan_readers_of_the_nuisance_block {

//// the nuisance values (inside NicoStan's gradient call: its nuisance block; outside it: the values set for use
//// outside NicoStan, whose gradient is not kept), checked for the values first..last (0-based, inclusive), and
//// the chunk run on this thread (both thread-local, so looked up once per read: in a model's shared library
//// every look-up of a thread-local variable is a call)
struct values_of_the_nuisance_block_t {
    stan::math::internal::nicostan_nuisance_block_t* block;
    const double* values;
    int size;
    stan::math::internal::nicostan_chunk_of_the_nuisance_block_t* chunk;
};

inline values_of_the_nuisance_block_t fn_values_of_the_nuisance_block(const char* reader, const int first,
                                                                        const int last) {

    stan::math::internal::nicostan_nuisance_block_t* block = stan::math::internal::nicostan_nuisance_block;
    const std::vector<double>& values_outside =
          stan::math::internal::nicostan_nuisance_values_used_outside_NicoStan;
    values_of_the_nuisance_block_t the_values{ block, (block != nullptr) ? block->values : values_outside.data(),
                                               (block != nullptr) ? block->size :
                                                                    static_cast<int>(values_outside.size()),
                                               &stan::math::internal::nicostan_chunk_of_the_nuisance_block };
    if (the_values.size == 0) {
        throw std::domain_error(std::string(reader) + ": no nuisance values (a model in NicoStan's "
                                "nuisance-block form is evaluated through NicoStan's gradient call, or after "
                                "nicostan_set_the_nuisance_values_used_outside_NicoStan_R)");
    }
    if (first < 0 || last < first - 1 || last >= the_values.size) {
        throw std::domain_error(std::string(reader) + ": values " + std::to_string(first + 1) + " to " +
                                std::to_string(last + 1) + " are outside the nuisance values (" +
                                std::to_string(the_values.size) + ")");
    }
    return the_values;

}

//// one value (0-based) of term "term" (0-based) as a node: of the chunk's own tape inside a chunk (with the
//// chunk's own values when the term is one of the chunk's own terms), else of the main tape
inline stan::math::var fn_node_of_value(const values_of_the_nuisance_block_t& the_values, const int value,
                                        const int term) {

    if (the_values.block == nullptr) return stan::math::var(the_values.values[value]);
    stan::math::internal::nicostan_chunk_of_the_nuisance_block_t& chunk = *the_values.chunk;
    if (!chunk.active) {
        stan::math::internal::nicostan_nuisance_block_t& block = *the_values.block;
        //// Only the model's own tape may reuse these leaves; a nested tape keeps its existing behaviour.
        if (stan::math::ChainableStack::instance_ == block.identity_of_the_main_tape &&
            stan::math::ChainableStack::instance_->nested_var_stack_sizes_.size() ==
            block.nested_depth_of_the_main_tape) {

            if (block.nodes_by_value_on_the_main_tape.empty()) {
                block.nodes_by_value_on_the_main_tape.resize(block.size, nullptr);
            }
            stan::math::vari*& existing_node = block.nodes_by_value_on_the_main_tape[value];
            if (existing_node != nullptr) return stan::math::var(existing_node);
            stan::math::var node(the_values.values[value]);
            block.nodes_on_the_main_tape.emplace_back(value, node.vi_);
            existing_node = node.vi_;
            return node;

        }
        stan::math::var node(the_values.values[value]);
        block.nodes_on_the_main_tape.emplace_back(value, node.vi_);
        return node;
    }
    auto* reuse = stan::math::internal::nicostan_reused_nuisance_nodes_of_this_map_job;
    const bool reuse_on_this_tape = reuse != nullptr && reuse->matches(the_values.block);
    stan::math::vari** reused_node = reuse_on_this_tape ? reuse->position_of_value(value) : nullptr;
    if (reused_node != nullptr && *reused_node != nullptr) return stan::math::var(*reused_node);
    stan::math::var node(the_values.values[value]);
    if (reused_node != nullptr) *reused_node = node.vi_;
    if (term >= chunk.first && term <= chunk.last) {
        chunk.nodes.emplace_back(value, node.vi_);
    } else {
        chunk.nodes_of_the_values_of_other_terms.emplace_back(value, node.vi_);
    }
    return node;

}

}  // namespace nicostan_readers_of_the_nuisance_block

template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, 1>
nicostan_nuisance_vector(const int& first_term, const int& last_term, const int& offset, const T_type& /*type*/,
                         std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = last_term - first_term + 1;
    const int first_value = offset + first_term - 1;
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_vector", first_value, first_value + n - 1);
    if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
        Eigen::Matrix<stan::math::var, -1, 1> values(n);
        for (int j = 0; j < n; ++j) {
            values.coeffRef(j) = readers::fn_node_of_value(the_values, first_value + j, first_term - 1 + j);
        }
        return values;
    } else {
        return Eigen::Map<const Eigen::VectorXd>(the_values.values + first_value, n);
    }

}

template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, 1>
nicostan_nuisance_vector_at(const std::vector<int>& terms, const int& offset, const T_type& /*type*/,
                            std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = static_cast<int>(terms.size());
    Eigen::Matrix<stan::return_type_t<T_type>, -1, 1> values(n);
    //// (the block once; each value's range checked on its own)
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_vector_at", 0, -1);
    for (int j = 0; j < n; ++j) {
        const int value = offset + terms[j] - 1;
        if (value < 0 || value >= the_values.size) {
            throw std::domain_error("nicostan_nuisance_vector_at: value " + std::to_string(value + 1) +
                                    " is outside the nuisance values (" + std::to_string(the_values.size) + ")");
        }
        if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
            values.coeffRef(j) = readers::fn_node_of_value(the_values, value, terms[j] - 1);
        } else {
            values.coeffRef(j) = the_values.values[value];
        }
    }
    return values;

}

template <typename T_type>
inline stan::return_type_t<T_type>
nicostan_nuisance_value(const int& term, const int& offset, const T_type& type,
                        std::ostream* pstream__ = nullptr) {

    return nicostan_nuisance_vector(term, term, offset, type, pstream__).coeff(0);

}

//// (before: through the vector reader, then a copy)
// template <typename T_type>
// inline std::vector<stan::return_type_t<T_type>>
// nicostan_nuisance_array_of_reals(const int& first_term, const int& last_term, const int& offset,
//                                  const T_type& type, std::ostream* pstream__ = nullptr) {
//     const auto values = nicostan_nuisance_vector(first_term, last_term, offset, type, pstream__);
//     return std::vector<stan::return_type_t<T_type>>(values.data(), values.data() + values.size());
// }
//// (the arrays are filled straight from the block, never through a vector and a copy)
template <typename T_type>
inline std::vector<stan::return_type_t<T_type>>
nicostan_nuisance_array_of_reals(const int& first_term, const int& last_term, const int& offset,
                                 const T_type& /*type*/, std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = last_term - first_term + 1;
    const int first_value = offset + first_term - 1;
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_array_of_reals", first_value,
                                                   first_value + n - 1);
    std::vector<stan::return_type_t<T_type>> values;
    if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
        values.reserve(n);
        for (int j = 0; j < n; ++j) {
            values.emplace_back(readers::fn_node_of_value(the_values, first_value + j, first_term - 1 + j));
        }
    } else {
        values.assign(the_values.values + first_value, the_values.values + first_value + n);
    }
    return values;

}

// template <typename T_type>
// inline std::vector<stan::return_type_t<T_type>>
// nicostan_nuisance_array_of_reals_at(const std::vector<int>& terms, const int& offset, const T_type& type,
//                                     std::ostream* pstream__ = nullptr) {
//     const auto values = nicostan_nuisance_vector_at(terms, offset, type, pstream__);
//     return std::vector<stan::return_type_t<T_type>>(values.data(), values.data() + values.size());
// }
template <typename T_type>
inline std::vector<stan::return_type_t<T_type>>
nicostan_nuisance_array_of_reals_at(const std::vector<int>& terms, const int& offset, const T_type& /*type*/,
                                    std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = static_cast<int>(terms.size());
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_array_of_reals_at", 0, -1);
    std::vector<stan::return_type_t<T_type>> values;
    values.reserve(n);
    for (int j = 0; j < n; ++j) {
        const int value = offset + terms[j] - 1;
        if (value < 0 || value >= the_values.size) {
            throw std::domain_error("nicostan_nuisance_array_of_reals_at: value " + std::to_string(value + 1) +
                                    " is outside the nuisance values (" + std::to_string(the_values.size) + ")");
        }
        if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
            values.emplace_back(readers::fn_node_of_value(the_values, value, terms[j] - 1));
        } else {
            values.push_back(the_values.values[value]);
        }
    }
    return values;

}

//// (an array of vectors or of row_vectors, values_per_term values per term)
template <int Rows, int Cols, typename T_type>
inline std::vector<Eigen::Matrix<stan::return_type_t<T_type>, Rows, Cols>>
nicostan_nuisance_array_of_vectors_or_row_vectors(const char* reader, const int first_term, const int last_term,
                                                  const int values_per_term, const int offset) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = last_term - first_term + 1;
    const int first_value = offset + (first_term - 1) * values_per_term;
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block(reader, first_value, first_value + n * values_per_term - 1);
    std::vector<Eigen::Matrix<stan::return_type_t<T_type>, Rows, Cols>> terms(n);
    for (int t = 0; t < n; ++t) {
        terms[t].resize(values_per_term);
        for (int j = 0; j < values_per_term; ++j) {
            const int value = first_value + t * values_per_term + j;
            if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
                terms[t].coeffRef(j) = readers::fn_node_of_value(the_values, value, first_term - 1 + t);
            } else {
                terms[t].coeffRef(j) = the_values.values[value];
            }
        }
    }
    return terms;

}

template <typename T_type>
inline std::vector<Eigen::Matrix<stan::return_type_t<T_type>, -1, 1>>
nicostan_nuisance_array_of_vectors(const int& first_term, const int& last_term, const int& values_per_term,
                                   const int& offset, const T_type& /*type*/,
                                   std::ostream* /*pstream__*/ = nullptr) {

    return nicostan_nuisance_array_of_vectors_or_row_vectors<-1, 1, T_type>(
                 "nicostan_nuisance_array_of_vectors", first_term, last_term, values_per_term, offset);

}

template <typename T_type>
inline std::vector<Eigen::Matrix<stan::return_type_t<T_type>, 1, -1>>
nicostan_nuisance_array_of_row_vectors(const int& first_term, const int& last_term, const int& values_per_term,
                                       const int& offset, const T_type& /*type*/,
                                       std::ostream* /*pstream__*/ = nullptr) {

    return nicostan_nuisance_array_of_vectors_or_row_vectors<1, -1, T_type>(
                 "nicostan_nuisance_array_of_row_vectors", first_term, last_term, values_per_term, offset);

}

//// (the terms as the rows of one matrix: the same values, nodes and order of the nodes as
////  nicostan_nuisance_array_of_row_vectors() then to_matrix(), without the row_vector of every term, each a
////  heap allocation and its free in every gradient, nor the copy into the matrix)
template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, -1>
nicostan_nuisance_matrix_of_terms(const int& first_term, const int& last_term, const int& values_per_term,
                                  const int& offset, const T_type& /*type*/,
                                  std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = last_term - first_term + 1;
    const int first_value = offset + (first_term - 1) * values_per_term;
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_matrix_of_terms", first_value,
                                                   first_value + n * values_per_term - 1);
    Eigen::Matrix<stan::return_type_t<T_type>, -1, -1> terms(n, values_per_term);
    //// (term by term, then value by value: the nodes made in the order of the array reader's)
    for (int t = 0; t < n; ++t) {
        for (int j = 0; j < values_per_term; ++j) {
            const int value = first_value + t * values_per_term + j;
            if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
                terms.coeffRef(t, j) = readers::fn_node_of_value(the_values, value, first_term - 1 + t);
            } else {
                terms.coeffRef(t, j) = the_values.values[value];
            }
        }
    }
    return terms;

}

//// (a block of one value per term, the values read by their own index: the first readers of the
////  nuisance-block form, kept with their meaning)
template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, 1>
nicostan_nuisance_values(const int& first, const int& last, const T_type& type,
                         std::ostream* pstream__ = nullptr) {

    return nicostan_nuisance_vector(first, last, 0, type, pstream__);

}

template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, 1>
nicostan_nuisance_values_at(const std::vector<int>& units, const T_type& type,
                            std::ostream* pstream__ = nullptr) {

    return nicostan_nuisance_vector_at(units, 0, type, pstream__);

}
//// (an array of matrices, rows x columns values per term: each term's matrix in Stan's own order of its values,
////  column by column, the terms one after the other)
template <typename T_type>
inline std::vector<Eigen::Matrix<stan::return_type_t<T_type>, -1, -1>>
nicostan_nuisance_array_of_matrices(const int& first_term,
                                    const int& last_term,
                                    const int& rows,
                                    const int& columns,
                                    const int& offset,
                                    const T_type& /*type*/,
                                    std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n = last_term - first_term + 1;
    const int values_per_term = rows * columns;
    const int first_value = offset + (first_term - 1) * values_per_term;
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_array_of_matrices", first_value,
                                                   first_value + n * values_per_term - 1);
    std::vector<Eigen::Matrix<stan::return_type_t<T_type>, -1, -1>> terms(n);
    for (int t = 0; t < n; ++t) {
        terms[t].resize(rows, columns);
        //// (column by column: Eigen's own order of a matrix's values, so value j of the term is coefficient j)
        for (int j = 0; j < values_per_term; ++j) {
            const int value = first_value + t * values_per_term + j;
            if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
                terms[t].coeffRef(j) = readers::fn_node_of_value(the_values, value, first_term - 1 + t);
            } else {
                terms[t].coeffRef(j) = the_values.values[value];
            }
        }
    }
    return terms;

}
//// (rows first_row..last_row and columns first_column..last_column of one matrix of the nuisance block, a matrix
////  of `rows` rows whose values start after `offset` values, column by column as Stan orders them: only the
////  values of these rows and columns become nodes, e.g. the rows of a chunk's own units, C_raw[c][start:end, ].
////  Every value counts as a value of another chunk's terms (term -1): its adjoint is added after all the chunks,
////  under the lock. Its row is not the term the other readers of the same block use (its index in the whole
////  block, or the matrix of an array of matrices), so a row counted as the chunk's own could also be counted as
////  their own by another chunk, and two threads would then add into the same value of the gradient at once)
template <typename T_type>
inline Eigen::Matrix<stan::return_type_t<T_type>, -1, -1>
nicostan_nuisance_matrix_block(const int& first_row,
                               const int& last_row,
                               const int& first_column,
                               const int& last_column,
                               const int& rows,
                               const int& offset,
                               const T_type& /*type*/,
                               std::ostream* /*pstream__*/ = nullptr) {

    namespace readers = nicostan_readers_of_the_nuisance_block;
    const int n_rows = last_row - first_row + 1;
    const int n_columns = last_column - first_column + 1;
    if (first_row < 1 || last_row > rows || n_rows < 0 || first_column < 1 || n_columns < 0) {
        throw std::domain_error("nicostan_nuisance_matrix_block: rows " + std::to_string(first_row) + " to " +
                                std::to_string(last_row) + " and columns " + std::to_string(first_column) +
                                " to " + std::to_string(last_column) + " are outside the matrix (" +
                                std::to_string(rows) + " rows)");
    }
    Eigen::Matrix<stan::return_type_t<T_type>, -1, -1> block_of_the_matrix(n_rows, n_columns);
    if (n_rows == 0 || n_columns == 0) return block_of_the_matrix;
    const int first_value = offset + (first_column - 1) * rows + (first_row - 1);
    const int last_value = offset + (last_column - 1) * rows + (last_row - 1);
    const readers::values_of_the_nuisance_block_t the_values =
          readers::fn_values_of_the_nuisance_block("nicostan_nuisance_matrix_block", first_value, last_value);
    for (int c = 0; c < n_columns; ++c) {
        for (int r = 0; r < n_rows; ++r) {
            const int value = offset + (first_column - 1 + c) * rows + (first_row - 1 + r);
            if constexpr (std::is_same<stan::return_type_t<T_type>, stan::math::var>::value) {
                block_of_the_matrix.coeffRef(r, c) = readers::fn_node_of_value(the_values, value, -1);
            } else {
                block_of_the_matrix.coeffRef(r, c) = the_values.values[value];
            }
        }
    }
    return block_of_the_matrix;

}
////
//// ---- the constraint of a nuisance block held by NicoStan: NicoStan holds the UNCONSTRAINED values of a
////      constrained block (e.g. matrix<lower=-7.5, upper=2.5>[n_studies, n_thr] C_raw), so each value the model
////      reads is constrained with Stan math's own transform, the one Stan's own parameters block uses
////      (stan::math::lb_constrain, ub_constrain, lub_constrain and offset_multiplier_constrain), and the
////      log-Jacobian of the transform is added to the target once per value, by
////      nicostan_log_jacobian_of_constraint() (the same transform, with Stan math's lp). The kind of the
////      constraint and its two bounds:
////          kind 1: lower = a (b not used);           kind 2: upper = b (a not used);
////          kind 3: lower = a and upper = b;          kind 4: offset = a and multiplier = b.
////      x is any of the types the readers return (real, vector, row_vector, matrix and arrays of these); a and b
////      are real numbers (data or parameters). Declared in the Stan functions block for each type read:
////          <type> nicostan_constrain(<type> x, int kind, real a, real b);
////          real nicostan_log_jacobian_of_constraint(<type> x, int kind, real a, real b);
////
template <typename T_x, typename T_a, typename T_b>
inline stan::math::promote_scalar_t<stan::return_type_t<T_x, T_a, T_b>, stan::plain_type_t<T_x>>
nicostan_constrain(const T_x& x,
                   const int& kind,
                   const T_a& a,
                   const T_b& b,
                   std::ostream* /*pstream__*/ = nullptr) {

    //// (the scalar type of x, a and b together: a transform that uses only one bound returns that of x and
    ////  that bound, promoted here)
    using T_return = stan::return_type_t<T_x, T_a, T_b>;
    if (kind == 1) return stan::math::promote_scalar<T_return>(stan::math::lb_constrain(x, a));
    if (kind == 2) return stan::math::promote_scalar<T_return>(stan::math::ub_constrain(x, b));
    if (kind == 3) return stan::math::promote_scalar<T_return>(stan::math::lub_constrain(x, a, b));
    if (kind == 4) return stan::math::promote_scalar<T_return>(stan::math::offset_multiplier_constrain(x, a, b));
    throw std::domain_error("nicostan_constrain: the kind of the constraint is " + std::to_string(kind) +
                            "; it must be 1 (lower), 2 (upper), 3 (lower and upper) or 4 (offset and "
                            "multiplier)");

}
template <typename T_x, typename T_a, typename T_b>
inline stan::return_type_t<T_x, T_a, T_b>
nicostan_log_jacobian_of_constraint(const T_x& x,
                                    const int& kind,
                                    const T_a& a,
                                    const T_b& b,
                                    std::ostream* /*pstream__*/ = nullptr) {

    //// (each transform with its own lp, of the type Stan math asks for; the constrained values are not kept)
    if (kind == 1) {
        stan::return_type_t<T_x, T_a> lp = 0;
        stan::math::lb_constrain(x, a, lp);
        return lp;
    }
    if (kind == 2) {
        stan::return_type_t<T_x, T_b> lp = 0;
        stan::math::ub_constrain(x, b, lp);
        return lp;
    }
    if (kind == 3) {
        stan::return_type_t<T_x, T_a, T_b> lp = 0;
        stan::math::lub_constrain(x, a, b, lp);
        return lp;
    }
    if (kind == 4) {
        stan::return_type_t<T_x, T_a, T_b> lp = 0;
        stan::math::offset_multiplier_constrain(x, a, b, lp);
        return lp;
    }
    throw std::domain_error("nicostan_log_jacobian_of_constraint: the kind of the constraint is " +
                            std::to_string(kind) + "; it must be 1 (lower), 2 (upper), 3 (lower and upper) or 4 "
                            "(offset and multiplier)");

}






















