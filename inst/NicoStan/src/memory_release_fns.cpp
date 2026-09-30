//// ===================================================================================================================================
// memory_release_fns.cpp
//
// ---- Release free heap memory to the operating system (used before the M_bytes/chain measurement) -----------------------------------
//
// After R's gc(), the glibc allocator keeps freed heap memory resident (up to 9 to 12 MB after the model and data are loaded). A
// gradient evaluated afterwards reuses those pages without growing the resident memory, so fn_estimate_initial_M_bytes_per_chain()
// calls this function in its child process before resetting the peak, so that the pages the gradient touches are counted.
//
// malloc_trim(0) returns every whole free page of every glibc malloc arena to the kernel (with MADV_DONTNEED). It is a glibc
// extension: on other C libraries (macOS, Windows, musl) this function does nothing and returns -1.
//

#include <Rcpp.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif




//// ---- release free heap memory ------------------------------------------------------------------------------------------------------
//
// Returns 1 if memory was released to the kernel, 0 if there was none to release, and -1 if malloc_trim() is not available
// (not glibc).
//
// [[Rcpp::export(rng = false)]]
int Rcpp_fn_release_free_heap_memory() {

        #if defined(__GLIBC__)
              return malloc_trim(0);
        #else
              return -1;
        #endif

}























