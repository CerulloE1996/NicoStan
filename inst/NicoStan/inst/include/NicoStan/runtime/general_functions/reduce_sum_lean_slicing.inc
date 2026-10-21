// NicoStan patch of stan/math/rev/functor/reduce_sum.hpp, "lean slicing" (applied by
// R_fn_bridge_memory_patch_reduce_sum_lean_slicing to NicoStan's private BridgeStan runtime when
// options(NicoStan.reduce_sum_lean_slicing = TRUE); never to a user's own Stan): reduce_sum's own bookkeeping of
// the sliced terms is done inside the chunks, in N_chunks blocks, instead of in passes over all N terms on the
// main thread. For every sliced term Stan's reduce_sum (1) writes a pointer to its node into a list on the main
// tape, (2) zero-fills its partial, both in passes over all N terms before the chunks, and (3) after the chunks
// adds the adjoints through the pointer list. Here:
//   (1) each chunk writes the pointers of its own terms into the list, from the same terms it is copying onto its
//       own tape (so they are already in cache), in place of the pass over all N on the main thread;
//   (2) each chunk zeroes its own partials just before it accumulates them (each term belongs to exactly one
//       chunk), so the separate zero-fill pass over all of them is gone;
//   (3) the adjoints are added by Stan's own precomputed_gradients_vari, through the same pointer list in the
//       same order, so the gradient is identical.
// General: any sliced argument (arrays of scalars, of vectors or row_vectors, of arrays; data or parameters) and
// any shared arguments. The same runtime for every math backend (Stan math, AVX2, AVX-512).
#include <algorithm>






















