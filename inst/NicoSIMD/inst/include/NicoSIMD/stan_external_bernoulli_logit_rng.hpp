#pragma once

#include "stan_external_scalar_functions.hpp"
#include <stan/math/prim/meta/VectorBuilder.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/scalar_seq_view.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <algorithm>

namespace nico::math {

// Keep the random generator and draw order; transform logits one chunk at a time.
template <typename T, class RNG>
inline typename stan::VectorBuilder<true, int, T>::type nico_bernoulli_logit_rng(
    const T& t,
    RNG& rng,
    std::ostream* pstream__) {

    using boost::bernoulli_distribution;
    using boost::variate_generator;
    stan::ref_type_t<T> t_ref = t;
    stan::math::check_finite("nico_bernoulli_logit_rng", "Logit transformed probability parameter", t_ref);
    stan::scalar_seq_view<T> t_vec(t_ref);
    const size_t n = stan::math::size(t);
    stan::VectorBuilder<true, int, T> output(n);

    if constexpr (std::is_arithmetic<std::decay_t<T>>::value) {

        const double probability = nico_inv_logit(t_ref, pstream__);
        variate_generator<RNG&, bernoulli_distribution<>> bernoulli_rng(
            rng, bernoulli_distribution<>(probability));
        output[0] = bernoulli_rng();

    } else {

        constexpr size_t chunk_size = 4096;
        Eigen::VectorXd logits(std::min(n, chunk_size));
        for (size_t first = 0; first < n; first += chunk_size) {

            const size_t count = std::min(chunk_size, n - first);
            logits.resize(count);
            for (size_t row = 0; row < count; ++row) {

                logits.coeffRef(row) = t_vec[first + row];

            }
            const Eigen::VectorXd probability = nico_inv_logit(logits, pstream__);
            for (size_t row = 0; row < count; ++row) {

                variate_generator<RNG&, bernoulli_distribution<>> bernoulli_rng(
                    rng, bernoulli_distribution<>(probability.coeff(row)));
                output[first + row] = bernoulli_rng();

            }

        }

    }
    return output.data();

}

}  // namespace nico::math

using nico::math::nico_bernoulli_logit_rng;
