#ifndef NICOSTAN_CHUNK_LOCAL_LEAPFROG_STEP_OF_THE_NUISANCE_BLOCK_HPP
#define NICOSTAN_CHUNK_LOCAL_LEAPFROG_STEP_OF_THE_NUISANCE_BLOCK_HPP

////
//// ---- The chunk-local leapfrog step of reduce_sum_chunked
////      (options(NicoStan.reduce_sum_chunk_local_leapfrog_step = TRUE), on top of
////      NicoStan.reduce_sum_nuisance_block and NicoStan.reduce_sum_lean_slicing): the part of
////      a leapfrog step that NicoStan's sampler makes on the nuisance values right after their gradient (the
////      closing half kick of the step, then the next step's half kick and rotation, or, after the last step,
////      the closing half kick alone) is made inside each chunk of the model's reduce_sum, on the chunk's own
////      values, whilst they and their gradient are still in the cache, instead of in a pass over all N values
////      after the gradient call. This description of the step is the one thing the sampler (NicoStan's own
////      code) and the model library (NicoStan's private BridgeStan runtime, reduce_sum_chunk_local_leapfrog_
////      step.inc) both see: the same text in both (the runtime patch inserts this file into the runtime's
////      reduce_sum.hpp). Plain C types only, so that it means the same in both.
////
////      The step itself is made by NicoStan's own code, through the function given here, so that every value is
////      made by the same compiled arithmetic as in NicoStan's pass over all the values; the model library only
////      says when (after the chunk's own adjoints) and for which values (the chunk's own values, but not those
////      read outside their own chunk, whose gradient is complete only after all the chunks). The values made
////      are written into separate arrays of the next values (never over the values the chunks read), and the
////      sampler takes them over after the gradient call.
////
struct nicostan_chunk_local_leapfrog_step_t {
    ////
    //// (set by NicoStan's sampler before the gradient call of a leapfrog step)
    void* NicoStan_pass_over_the_values;
    void (*fn_make_the_step_of_the_values_first_to_end)(void* NicoStan_pass_over_the_values, int first, int end);
    const int* values_read_outside_their_own_chunk_in_the_previous_gradient_call;   // ascending, each once
    int n_values_read_outside_their_own_chunk_in_the_previous_gradient_call;
    ////
    //// (set by the model library during the gradient call; the arrays are the library's own, valid until the
    ////  next gradient call on the same thread)
    int step_made_inside_the_chunks;                     // 1 when the gradient call made the step inside them
    const int* first_and_end_of_the_runs_of_values_stepped_inside_the_chunks;   // ascending, 2 values per run
    int n_runs_of_values_stepped_inside_the_chunks;
    const int* values_read_outside_their_own_chunk;     // in this gradient call: ascending, each once
    int n_values_read_outside_their_own_chunk;
};

#ifndef NICOSTAN_CHUNK_LOCAL_LEAPFROG_STEP_INSIDE_THE_MODEL_LIBRARY

////
//// ---- (NicoStan's side) the step of the gradient call being made on this thread: set by the leapfrog loop of
////      the sampler only, for the duration of the gradient call of each leapfrog step, so that no other gradient
////      call (the initial gradient, the lp_grad timings, the Hessian adaptation, ...) ever makes it.
////
inline thread_local nicostan_chunk_local_leapfrog_step_t*
      nicostan_chunk_local_leapfrog_step_of_this_gradient_call = nullptr;

struct nicostan_scope_of_the_chunk_local_leapfrog_step_of_a_gradient_call {
    nicostan_chunk_local_leapfrog_step_t* previous_step;
    explicit nicostan_scope_of_the_chunk_local_leapfrog_step_of_a_gradient_call(
                   nicostan_chunk_local_leapfrog_step_t* step)
          : previous_step(nicostan_chunk_local_leapfrog_step_of_this_gradient_call) {

        nicostan_chunk_local_leapfrog_step_of_this_gradient_call = step;

    }
    ~nicostan_scope_of_the_chunk_local_leapfrog_step_of_a_gradient_call() {

        nicostan_chunk_local_leapfrog_step_of_this_gradient_call = previous_step;

    }
};

#endif

#endif























