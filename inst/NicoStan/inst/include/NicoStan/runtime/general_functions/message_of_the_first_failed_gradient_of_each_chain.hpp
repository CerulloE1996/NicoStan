#pragma once

#include <cstdio>
#include <exception>

////
//// ---- The message of the first failed gradient of each chain (a diagnostic; it changes nothing the sampler does):
////      a gradient that throws (e.g. "Gradient computation failed: <BridgeStan / Stan error message>") is still
////      recorded as a divergent transition and its proposal rejected, as before; this only writes the message of
////      the FIRST such failure of each chain thread to stderr, with the chain and the iteration, so that the cause
////      of a "divergence" that is in fact a failed gradient can be read in the log of the run.
////      The chain and the iteration of the chain thread are set at the start of every iteration
////      (fn_sample_HMC_multi_iter_single_thread); each chain runs on a thread of its own.
////
inline thread_local int chain_of_this_thread_for_the_message_of_a_failed_gradient = -1;
inline thread_local int iteration_of_this_thread_for_the_message_of_a_failed_gradient = -1;
inline thread_local int n_failed_gradients_of_this_thread = 0;

inline void fn_set_the_chain_and_the_iteration_of_this_thread_for_the_message_of_a_failed_gradient(
      const int chain,
      const int iteration) {

      if (chain != chain_of_this_thread_for_the_message_of_a_failed_gradient) n_failed_gradients_of_this_thread = 0;
      chain_of_this_thread_for_the_message_of_a_failed_gradient = chain;
      iteration_of_this_thread_for_the_message_of_a_failed_gradient = iteration;

}

inline void fn_write_the_message_of_the_first_failed_gradient_of_this_chain(const char *where_it_failed,
                                                                            const char *message) noexcept {

      n_failed_gradients_of_this_thread += 1;
      if (n_failed_gradients_of_this_thread != 1) return;
      std::fprintf(stderr,
                   "NicoStan: the first failed gradient of chain %d, at iteration %d (%s), recorded as a "
                   "divergent transition: %s\n",
                   chain_of_this_thread_for_the_message_of_a_failed_gradient,
                   iteration_of_this_thread_for_the_message_of_a_failed_gradient,
                   where_it_failed,
                   (message != nullptr) ? message : "no message");
      std::fflush(stderr);

}























