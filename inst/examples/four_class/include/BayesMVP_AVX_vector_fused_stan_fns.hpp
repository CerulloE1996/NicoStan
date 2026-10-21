#pragma once

// Optional extension: real whole-vector autodiff and ordinary normal-interval fusion.
// The existing header, mathematical kernels and AoS overloads remain unchanged.
// A single vector-valued node owns contiguous values AND adjoints. Keep the caller in this
// representation across operations; converting back to Matrix<var> at every call defeats it.
// Plain vector-valued vars are contiguous. Also support strided column views safely.
// namespace bmvp_stan_ext
/*pstream__*/
// Return N x 2: column 1 is z, column 2 log(probability). One callback; never fake
// scalar nodes or omit their normal registration/reset. AoS scatter is ordered for aliases.
// Ordinary interval only. The Stan caller retains its existing selection of stable
// tails/narrow intervals/Phi_type=2. Do not clamp probabilities or change that selection.
// At u extremely close to 0/1, the original reverse graph has cancellation of
// very large terms. Preserve that graph at these exceptional inputs; do not claim
// floating-point equivalence of an algebraically regrouped derivative there.
// outcome is a discrete 0/1 mask, never differentiated. stanc may promote a
// locally indexed data vector to Matrix<var>; its values still define the mask.
// Preserve the existing model's expression/order rather than simplify away numerical terms.
// namespace bmvp_stan_ext

#include <NicoSIMD/stan_external_vector_fused_functions.hpp>






















