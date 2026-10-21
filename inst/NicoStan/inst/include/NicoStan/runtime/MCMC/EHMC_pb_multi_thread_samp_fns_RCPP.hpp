#pragma once

 
 
#include <filesystem>
 

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tbb/concurrent_vector.h>

 
 
#include <chrono> 
#include <unordered_map>
#include <memory>
#include <thread>
#include <functional>

 
 
//// ANSI codes for different colors
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

 
using namespace Rcpp;
 
 
 
 
 




#include <tbb/task_scheduler_observer.h>
#include <tbb/task_arena.h>
#include <NicoStan/runtime/general_functions/full_clock_of_the_compute_threads.hpp>
 
 class PinningObserver : public tbb::task_scheduler_observer {
   
       std::vector<int> core_ids;
   
     public:
       PinningObserver(const std::vector<int>& cores) 
         : core_ids(cores) {

         observe(true); // Activate the observer

       }
       
       void on_scheduler_entry(bool worker) override {
         int thread_id = tbb::this_task_arena::current_thread_index();
         if (thread_id < core_ids.size()) {
           cpu_set_t cpuset;
           CPU_ZERO(&cpuset);
           CPU_SET(core_ids[thread_id], &cpuset);
           pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
         }
       }
   
 };
 
 
 
 
 

// --------------------------------- RcpParallel  functions  -- SAMPLING fn --------------------------------------------------------------------------------



#include <atomic>
#include <chrono>


 
 
//// (the steady clock's seconds, for the chains' start of timed iterations)
inline double fn_seconds_on_steady_clock() {

      return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

}

//// the gradient at a chain's starting state, evaluated before the chain's timed iterations start, so that
//// the first timed iteration reuses it, as every later iteration reuses the gradient at its current state:
//// a timed call of n iterations at L leapfrog steps then evaluates exactly n L gradients inside its timed
//// window (before, n L + 1: the first iteration evaluated the gradient at the starting state inside the
//// window). It is not an iteration: no leapfrog step, no draw, no change of the chain's state, so the draws
//// are the same. Only for the joint diffusion sampler with a nuisance block (the only sampler that reuses
//// the gradient at the current state); the flag stays false, and the first iteration evaluates the gradient
//// as before, when it is already valid (after untimed iterations), when the evaluation throws or when it is
//// not finite.
template <typename Fn_lp_grad_at_the_current_state_type>
inline void fn_gradient_at_the_starting_state_before_the_timed_iterations(
        bool &gradient_at_the_current_state_is_valid,
        const bool the_sampler_reuses_the_gradient_at_the_current_state,
        Fn_lp_grad_at_the_current_state_type &fn_lp_grad_at_the_current_state,
        const Eigen::Ref<const Eigen::Matrix<double, -1, 1>> lp_and_grad_outs) {

      if (gradient_at_the_current_state_is_valid) return;
      if (!the_sampler_reuses_the_gradient_at_the_current_state) return;
      try {
            fn_lp_grad_at_the_current_state();
      } catch (...) {
            return;
      }
      if (lp_and_grad_outs.allFinite()) gradient_at_the_current_state_is_valid = true;

}

//// the untimed iterations before a chain's timed iterations, at their own leapfrog steps (1
//// step is enough to start the threads, build the tapes and touch the memory): the chain's trajectory lengths
//// set to that many steps of its step sizes for the untimed iterations only, then put back for the timed ones
//// (L_of_the_untimed_iterations_before_timed_iterations = 0: the untimed iterations as the timed ones)
template <typename Args_type, typename Fn_run_iterations_type>
inline void fn_run_untimed_iterations_at_their_own_L(Args_type &EHMC_args_of_the_chain,
                                                     const int n_untimed_iterations,
                                                     Fn_run_iterations_type &fn_run_iterations) {

      const int L_of_the_untimed_iterations =
            EHMC_args_of_the_chain.L_of_the_untimed_iterations_before_timed_iterations;
      if (L_of_the_untimed_iterations <= 0) {
            fn_run_iterations(n_untimed_iterations);
            return;
      }
      const double tau_main_of_the_timed_iterations = EHMC_args_of_the_chain.tau_main;
      const double tau_us_of_the_timed_iterations = EHMC_args_of_the_chain.tau_us;
      EHMC_args_of_the_chain.tau_main = L_of_the_untimed_iterations * EHMC_args_of_the_chain.eps_main;
      EHMC_args_of_the_chain.tau_us = L_of_the_untimed_iterations * EHMC_args_of_the_chain.eps_us;
      fn_run_iterations(n_untimed_iterations);
      EHMC_args_of_the_chain.tau_main = tau_main_of_the_timed_iterations;
      EHMC_args_of_the_chain.tau_us = tau_us_of_the_timed_iterations;

}

//// ---- The Stan model shared by the chains, kept between sampling calls of the same library, data file and
////      seed: building it reads and parses the data file, and at N = 50,000 this took 0.76 s of every
////      sampling call (JLS; 0.01 s at N = 500), outside the chains' timed iterations but on every call of a
////      layout timing.
////      It is built again only when the library, the data file (its path, size or time of modification) or the
////      seed changes; the model before it is destroyed then. The data and transformed data are read-only, so a
////      model kept between calls gives every call the same values as a model built for it.
//// ---- Not one model but a short list of them, one per data given: a layout timing changes
////      only chunk_size in the data, and the search of the layouts comes back to the same chunk_size at every
////      number of threads per chain and at every candidate of the seconds per timed call; with one kept model
////      every such layout parsed the data again (6.5 to 8.5 s of a 37 MB data file at N = 50,000). A model is
////      found again by its library (path, size, time), its seed and its data: a data file by path, size and
////      time; the data's JSON text itself (a string that does not end in .json) by the text. At most
////      n_Stan_models_kept_between_sampling_calls models are kept; the one used longest ago is destroyed
////      first. Every kept model is read-only after it is built, so each call gets the same values as from a
////      model built for it.
struct Stan_model_kept_between_sampling_calls_t {
      std::string model_so_file;
      std::string json_file;
      unsigned int seed = 0;
      std::uintmax_t size_of_the_library = 0;
      std::uintmax_t size_of_the_data_file = 0;
      std::filesystem::file_time_type modification_time_of_the_library{};
      std::filesystem::file_time_type modification_time_of_the_data_file{};
      Stan_model_struct model{};
      bool is_built = false;
      unsigned long long last_used_at_call = 0;
      //// (the burn-in workers alive that use this model: a model in use is never destroyed to make room)
      int n_burn_in_workers_using_it = 0;
};

// inline constexpr std::size_t n_Stan_models_kept_between_sampling_calls = 10;
//// (24: the burn-in timings come back to each of the 11 or so chunk sizes of an N at every number of chains and
////  threads per chain, and with fewer kept models than chunk sizes the model used longest ago would always be
////  the next one needed, i.e. each would be built again every time)
inline constexpr std::size_t n_Stan_models_kept_between_sampling_calls = 24;

inline std::vector<Stan_model_kept_between_sampling_calls_t>& fn_Stan_models_kept_between_sampling_calls() {

      static std::vector<Stan_model_kept_between_sampling_calls_t> kept;
      return kept;

}

//// (the one kept model of the earlier version, for the code that still reads it: the most recently used)
inline Stan_model_kept_between_sampling_calls_t& fn_Stan_model_kept_between_sampling_calls() {

      std::vector<Stan_model_kept_between_sampling_calls_t> &kept = fn_Stan_models_kept_between_sampling_calls();
      static Stan_model_kept_between_sampling_calls_t none;
      if (kept.empty()) return none;
      std::size_t index_of_the_most_recent = 0;
      for (std::size_t i = 1; i < kept.size(); ++i) {
            if (kept[i].last_used_at_call > kept[index_of_the_most_recent].last_used_at_call) {
                  index_of_the_most_recent = i;
            }
      }
      return kept[index_of_the_most_recent];

}

inline bool fn_json_file_is_a_path(const std::string &json_file) {

      const std::string ending = ".json";
      return (json_file.size() >= ending.size()) &&
             (json_file.compare(json_file.size() - ending.size(), ending.size(), ending) == 0);

}

inline Stan_model_struct fn_Stan_model_of_this_sampling_call(const std::string &model_so_file,
                                                             const std::string &json_file,
                                                             const unsigned int seed) {

      std::vector<Stan_model_kept_between_sampling_calls_t> &kept = fn_Stan_models_kept_between_sampling_calls();
      static unsigned long long n_calls = 0;
      ++n_calls;
      std::error_code error_of_the_library;
      std::error_code error_of_the_data_file;
      const std::uintmax_t size_of_the_library = std::filesystem::file_size(model_so_file, error_of_the_library);
      const std::filesystem::file_time_type modification_time_of_the_library =
            std::filesystem::last_write_time(model_so_file, error_of_the_library);
      const bool the_data_is_a_file = fn_json_file_is_a_path(json_file);
      std::uintmax_t size_of_the_data_file = 0;
      std::filesystem::file_time_type modification_time_of_the_data_file{};
      if (the_data_is_a_file) {
            size_of_the_data_file = std::filesystem::file_size(json_file, error_of_the_data_file);
            modification_time_of_the_data_file =
                  std::filesystem::last_write_time(json_file, error_of_the_data_file);
      }
      //// (a file that cannot be read for its size or time is never taken as unchanged: the model is built again)
      if (!error_of_the_library && !(the_data_is_a_file && error_of_the_data_file)) {
            for (std::size_t i = 0; i < kept.size(); ++i) {
                  const bool the_same_model =
                        kept[i].is_built &&
                        (kept[i].model_so_file == model_so_file) &&
                        (kept[i].seed == seed) &&
                        (kept[i].size_of_the_library == size_of_the_library) &&
                        (kept[i].modification_time_of_the_library == modification_time_of_the_library) &&
                        (kept[i].json_file == json_file) &&
                        (!the_data_is_a_file ||
                         ((kept[i].size_of_the_data_file == size_of_the_data_file) &&
                          (kept[i].modification_time_of_the_data_file == modification_time_of_the_data_file)));
                  if (the_same_model) {
                        kept[i].last_used_at_call = n_calls;
                        return kept[i].model;
                  }
            }
      }
      //// (not kept: the model used longest ago makes room when the list is full, then this one is built)
      // if (kept.size() >= n_Stan_models_kept_between_sampling_calls) {
      //       std::size_t index_of_the_oldest = 0;
      //       for (std::size_t i = 1; i < kept.size(); ++i) {
      //             if (kept[i].last_used_at_call < kept[index_of_the_oldest].last_used_at_call) {
      //                   index_of_the_oldest = i;
      //             }
      //       }
      //       if (kept[index_of_the_oldest].is_built) {
      //             fn_bs_destroy_Stan_model(kept[index_of_the_oldest].model);
      //       }
      //       kept.erase(kept.begin() + static_cast<std::ptrdiff_t>(index_of_the_oldest));
      // }
      //// (only a model that no burn-in worker alive uses can make room; when every kept model is in use, the
      ////  list grows by one instead)
      if (kept.size() >= n_Stan_models_kept_between_sampling_calls) {
            bool a_model_can_make_room = false;
            std::size_t index_of_the_oldest = 0;
            for (std::size_t i = 0; i < kept.size(); ++i) {
                  if (kept[i].n_burn_in_workers_using_it > 0) continue;
                  if (!a_model_can_make_room ||
                      (kept[i].last_used_at_call < kept[index_of_the_oldest].last_used_at_call)) {

                        index_of_the_oldest = i;
                        a_model_can_make_room = true;

                  }
            }
            if (a_model_can_make_room) {
                  if (kept[index_of_the_oldest].is_built) {
                        fn_bs_destroy_Stan_model(kept[index_of_the_oldest].model);
                  }
                  kept.erase(kept.begin() + static_cast<std::ptrdiff_t>(index_of_the_oldest));
            }
      }
      Stan_model_kept_between_sampling_calls_t built;
      built.model = fn_load_Stan_model_and_data(model_so_file, json_file, seed);
      built.model_so_file = model_so_file;
      built.json_file = json_file;
      built.seed = seed;
      built.size_of_the_library = size_of_the_library;
      built.size_of_the_data_file = size_of_the_data_file;
      built.modification_time_of_the_library = modification_time_of_the_library;
      built.modification_time_of_the_data_file = modification_time_of_the_data_file;
      built.is_built = true;
      built.last_used_at_call = n_calls;
      kept.push_back(built);
      return kept.back().model;

}


//// (a burn-in worker counts itself as a user of its kept model when it is built (change 1) and stops when it is
////  destroyed (change -1); a model not kept is not counted)
inline void fn_count_a_burn_in_worker_using_a_kept_Stan_model(const bs_model *bs_model_ptr, const int change) {

      if (bs_model_ptr == nullptr) return;
      std::vector<Stan_model_kept_between_sampling_calls_t> &kept = fn_Stan_models_kept_between_sampling_calls();
      for (std::size_t i = 0; i < kept.size(); ++i) {
            if (kept[i].is_built && (kept[i].model.bs_model_ptr == bs_model_ptr)) {
                  kept[i].n_burn_in_workers_using_it = std::max(0, kept[i].n_burn_in_workers_using_it + change);
                  return;
            }
      }

}


class RcppParallel_EHMC_sampling : public RcppParallel::Worker {

public:

          //////////////////// ---- declare variables
          // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
          //       dqrng::xoshiro256plus global_rng_main;
          //       dqrng::xoshiro256plus global_rng_nuisance;
          // #endif
          
          const uint64_t global_seed;
          const int n_threads;
          const int n_iter;
          const bool partitioned_HMC;
          const bool diffusion_HMC;
          const std::string Model_type;
          const bool sample_nuisance;
          const bool force_autodiff;
          const bool force_PartialLog;
          const bool multi_attempts;
          
          //// ---- local storage:
          std::vector<HMC_output_single_chain> HMC_outputs;
          std::vector<HMCResult> HMC_inputs;
          ////
          // std::vector<LC_MVP_workspace_struct> LC_MVP_ws_structs;
          std::vector<std::vector<LC_MVP_workspace_struct>> LC_MVP_ws_structs_double_vec;
          
          const int n_threads_WCP;
          
          //// ---- Input data (to read)
          const Eigen::Matrix<double, -1, -1>  &theta_main_vectors_all_chains_input_from_R_RcppPar;
          const Eigen::Matrix<double, -1, -1>  &theta_us_vectors_all_chains_input_from_R_RcppPar;
          
          //// data:
          const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies;
          // const Eigen::Matrix<int, -1, -1> &y_copy;
          
          //// ---- input structs:
          const std::vector<Model_fn_args_struct>   &Model_args_as_cpp_struct_copies; 
          ////
          std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies;  //// This has to be modifiable
          ////
          const std::vector<EHMC_Metric_struct> &EHMC_Metric_as_cpp_struct_copies;
          
          //////////////////// ---- declare SAMPLING-SPECIFIC variables:
          //// ---- references to R trace matrices:
          const int n_nuisance_to_track;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs;
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance;
          //// ---- this only gets used for built-in models, for Stan models log_lik must be defined in the "transformed parameters" block.
          std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik;
          
          ////
          const bool use_disk;
          const std::string trace_dir;
          
          //// ---- per-chain wall-clock start / end (seconds on the steady clock), for the timing line printed by
          ////      Rcpp_fn_RcppParallel_EHMC_sampling (each chain writes only its own entry):
          std::vector<double> chain_start_seconds;
          std::vector<double> chain_end_seconds;
          //// the start of each chain's timed iterations (after its untimed ones; the start of its iterations
          //// when there are none), for the timing of a layout in one call:
          std::vector<double> chain_timed_start_seconds;
          //// the end of each chain's timed iterations, taken straight after them, BEFORE the release of the Stan
          //// model's memory, so that the timed seconds are those of the iterations only:
          std::vector<double> chain_timed_end_seconds;
          //// the Stan model (its data and transformed data) of every chain: built ONCE, in the constructor, with
          //// the seed of chain 0, and used by every chain (BridgeStan allows concurrent calls of one model from
          //// several threads, with STAN_THREADS; the data are read-only), so that the process holds one copy of
          //// the data rather than one per chain (at large N, the copies of every chain filled the shared cache);
          //// kept between sampling calls (fn_Stan_model_of_this_sampling_call()), not destroyed with the worker:
          Stan_model_struct Stan_model_shared_by_the_chains;

  ////////////// ---- Constructor (initialise these with the SOURCE format)
  RcppParallel_EHMC_sampling(  const int  &n_threads_R,
                               const uint64_t  &global_seed_R,
                               const int  &n_iter_R,
                               const bool &partitioned_HMC_R,
                               const bool &diffusion_HMC_R,
                               const std::string &Model_type_R,
                               const bool &sample_nuisance_R,
                               const bool &force_autodiff_R,
                               const bool &force_PartialLog_R,
                               const bool &multi_attempts_R,
                               //// ---- inputs
                               const Eigen::Matrix<double, -1, -1> &theta_main_vectors_all_chains_input_from_R,
                               const Eigen::Matrix<double, -1, -1> &theta_us_vectors_all_chains_input_from_R,
                               //// ---- data:
                               const std::vector<Eigen::Matrix<int, -1, -1>> &y_copies_,
                               // const Eigen::Matrix<int, -1, -1> &y_copy_,
                               //// ---- input structs:
                               const std::vector<Model_fn_args_struct>  &Model_args_as_cpp_struct_copies_,   // READ-ONLY
                               std::vector<EHMC_fn_args_struct> &EHMC_args_as_cpp_struct_copies_,  
                               const std::vector<EHMC_Metric_struct>  &EHMC_Metric_as_cpp_struct_copies_, // READ-ONLY
                               //// ---- For POST-BURNIN only:
                               const int &n_nuisance_to_track_R,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_divs_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_nuisance_,
                               std::vector<Eigen::Matrix<double, -1, -1>> &trace_output_log_lik_,
                               ////
                               const bool &use_disk_,
                               const std::string &trace_dir_,
                               ////
                               const int &n_threads_WCP_
                               )
    :
    n_threads(n_threads_R),
    global_seed(global_seed_R),
    n_iter(n_iter_R),
    partitioned_HMC(partitioned_HMC_R),
    diffusion_HMC(diffusion_HMC_R),
    Model_type(Model_type_R),
    sample_nuisance(sample_nuisance_R),
    force_autodiff(force_autodiff_R),
    force_PartialLog(force_PartialLog_R),
    multi_attempts(multi_attempts_R) ,
    //// ---- inputs:
    theta_main_vectors_all_chains_input_from_R_RcppPar(theta_main_vectors_all_chains_input_from_R),
    theta_us_vectors_all_chains_input_from_R_RcppPar(theta_us_vectors_all_chains_input_from_R),
    //// ---- data:
    y_copies(y_copies_),
    // y_copy(y_copy_),
    //// ---- input structs:
    Model_args_as_cpp_struct_copies(Model_args_as_cpp_struct_copies_), // bookmark
    EHMC_args_as_cpp_struct_copies(EHMC_args_as_cpp_struct_copies_),
    EHMC_Metric_as_cpp_struct_copies(EHMC_Metric_as_cpp_struct_copies_), // bookmark
    //// ---- For POST-BURNIN only:
    n_nuisance_to_track(n_nuisance_to_track_R), 
    trace_output(trace_output_),
    trace_output_divs(trace_output_divs_),
    trace_output_nuisance(trace_output_nuisance_),
    trace_output_log_lik(trace_output_log_lik_),
    ////
    use_disk(use_disk_),
    trace_dir(trace_dir_),
    ////
    n_threads_WCP(n_threads_WCP_)
    {
          // #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
          //     global_rng_main.seed(global_seed_R);
          //     global_rng_nuisance.seed(global_seed_R + 1e6);
          // #endif
          
          const int N = Model_args_as_cpp_struct_copies[0].N;
          const int n_nuisance =  Model_args_as_cpp_struct_copies[0].n_nuisance;
          const int n_params_main = Model_args_as_cpp_struct_copies[0].n_params_main;
          
          // const bool use_disk = true;  // could pass this as a parameter instead
          // const std::string trace_dir = "/tmp/hmc_traces";
          
          std::cout << "Msg #1 (from RcppParallel_EHMC_sampling class) - start" << std::endl;
          
          HMC_outputs.reserve(n_threads_R);
          HMC_inputs.reserve(n_threads_R);
          chain_start_seconds.assign(n_threads_R, 0.0);
          chain_end_seconds.assign(n_threads_R, 0.0);
          chain_timed_start_seconds.assign(n_threads_R, 0.0);
          chain_timed_end_seconds.assign(n_threads_R, 0.0);
          ////
          //// log-lik trace rows: N, or 0 when the R side said nothing will read it (store_log_lik_trace = FALSE):
          const int n_log_lik_rows = EHMC_args_as_cpp_struct_copies_[0].store_log_lik_trace ? N : 0;
          ////
          //// This loop runs SERIALLY for every chain before parallelFor, so it must not zero-fill large buffers:
          //// HMC_inputs[i] (~5 MB per chain at n_nuisance = 60,000) is sized and zero-filled inside the chain's own
          //// thread at the start of operator() instead - same values, done in parallel.
          for (int i = 0; i < n_threads_R; ++i) {
                HMC_outputs.emplace_back( n_iter_R, n_nuisance_to_track, n_params_main, n_nuisance, N,
                                          use_disk, i, trace_dir, n_log_lik_rows);
                HMC_inputs.emplace_back( 0, 0, 0);
          }
          ////
          //// the Stan model of every chain (built once, with the seed chain 0 used to build its own model):
          if (Model_type == "Stan") {
                const int seed_of_chain_0 = fn_chain_seed_int(static_cast<std::uint64_t>(global_seed_R), 0, 0);
                // Stan_model_shared_by_the_chains =
                //       fn_load_Stan_model_and_data( Model_args_as_cpp_struct_copies_[0].model_so_file,
                //                                    Model_args_as_cpp_struct_copies_[0].json_file_path,
                //                                    seed_of_chain_0);
                //// (the model kept between sampling calls: built again only when the library, the data file or
                ////  the seed changes)
                Stan_model_shared_by_the_chains =
                      fn_Stan_model_of_this_sampling_call( Model_args_as_cpp_struct_copies_[0].model_so_file,
                                                           Model_args_as_cpp_struct_copies_[0].json_file_path,
                                                           seed_of_chain_0);
          }
          
          // // LC_MVP_ws_structs.resize(n_threads_R);
          // LC_MVP_ws_structs.reserve(n_threads_R);
          
          LC_MVP_ws_structs_double_vec.resize(n_threads_R);
          
          if ((Model_type == "LC_MVP") || (Model_type == "MVP") || (Model_type == "LC_MVOP") || (Model_type == "MVOP")) {
            
                  const int n_tests = y_copies_[0].cols();
                  const int n_class = Model_args_as_cpp_struct_copies_[0].Model_args_ints(1);
                  const int n_chunks = Model_args_as_cpp_struct_copies_[0].Model_args_ints(3);
                  const std::string &vect_type = Model_args_as_cpp_struct_copies_[0].Model_args_strings(0);
                  const Eigen::Matrix<int, -1, -1> &n_covariates_per_outcome_vec = Model_args_as_cpp_struct_copies_[0].Model_args_mats_int[0];
                  
                  const int n_corrs = n_class * n_tests * (n_tests - 1) / 2;
                  
                  int n_covariates_total, n_covariates_max;
                  if (n_class > 1) {
                    const int n_covariates_total_nd = n_covariates_per_outcome_vec.row(0).sum();
                    const int n_covariates_total_d  = n_covariates_per_outcome_vec.row(1).sum();
                    n_covariates_total = n_covariates_total_nd + n_covariates_total_d;
                    const int n_covariates_max_nd = n_covariates_per_outcome_vec.row(0).maxCoeff();
                    const int n_covariates_max_d  = n_covariates_per_outcome_vec.row(1).maxCoeff();
                    n_covariates_max = std::max(n_covariates_max_nd, n_covariates_max_d);
                  } else {
                    n_covariates_total = n_covariates_per_outcome_vec.sum();
                    n_covariates_max = n_covariates_per_outcome_vec.array().maxCoeff();
                  }
                  
                  int vec_size;
                  if      (vect_type == "AVX512") vec_size = 8;
                  else if (vect_type == "AVX2")   vec_size = 4;
                  else if (vect_type == "AVX")    vec_size = 2;
                  else                            vec_size = 1;
                  
                  ChunkSizeInfo chunk_info = calculate_chunk_sizes( N, 
                                                                    vec_size, 
                                                                    n_chunks);
                  
                  for (int i = 0; i < n_threads_R; ++i) {
                    
                      LC_MVP_ws_structs_double_vec[i].reserve(n_threads_WCP_);
                      
                      for (int t = 0; t < n_threads_WCP_; ++t) {
                        
                          LC_MVP_ws_structs_double_vec[i].emplace_back(  chunk_info.normal_chunk_size,
                                                                         n_tests,
                                                                         n_class,
                                                                         n_covariates_max,
                                                                         n_corrs,
                                                                         n_covariates_total);
                          
                          // int padded_chunk = ((chunk_info.normal_chunk_size + vec_size - 1) / vec_size) * vec_size;
                          // LC_MVP_ws_structs_double_vec[i].emplace_back(  padded_chunk, 
                          //                                                n_tests, 
                          //                                                n_class,
                          //                                                n_covariates_max,
                          //                                                n_corrs, 
                          //                                                n_covariates_total);
                      }
                  
                  }
                  
                  std::cout << "Pre-allocated " << n_threads_R << " lp_grad workspaces ("
                            << "chunk_size=" << chunk_info.normal_chunk_size 
                            << ", n_tests=" << n_tests << ")" << std::endl;
            
          } // else: LC_MVP_ws_structs[i].is_allocated stays false (dummy)
          
          //// (debug prints removed: the previous unconditional "nuisance file not
          //// open" message fired on every run, including sample_nuisance = FALSE,
          //// and was noise - the actual file state is checked in store_iteration()
          //// only when a nuisance write actually happens.)
          //// (every thread of the process, TBB's workers included, asks for the full clock, once per sampling
          ////  call: see full_clock_of_the_compute_threads.hpp)
          fn_ask_the_clock_governor_for_the_full_clock_of_every_thread_of_this_process();
    }

  ////////////// RcppParallel Parallel operator
  ////////////// ---- Destructor: the Stan model shared by the chains, once every chain has finished
  ~RcppParallel_EHMC_sampling() {

          // if (Stan_model_shared_by_the_chains.bs_model_ptr != nullptr) {
          //       fn_bs_destroy_Stan_model(Stan_model_shared_by_the_chains);
          // }
          //// (the model is kept between sampling calls, and destroyed by fn_Stan_model_of_this_sampling_call()
          ////  when a call needs another library, data file or seed)

  }

  void operator() (std::size_t begin, std::size_t end) {
           
              //// (this chain's thread asks for the full clock, once: full_clock_of_the_compute_threads.hpp)
              fn_ask_the_clock_governor_for_the_full_clock_whenever_this_thread_runs();
              const uint64_t global_seed_main =     global_seed;
              const uint64_t global_seed_nuisance = global_seed + 1e6;
              const int global_seed_main_int =      static_cast<int>(global_seed_main);
              const int global_seed_nuisance_int =  static_cast<int>(global_seed_nuisance);
      
              //// Process all chains from begin to end:
              for (std::size_t i = begin; i < end; ++i) {
                
                           // OPEN FILE HERE - inside the parallel loop, per chain
                           if (use_disk == true) {
                                 
                                 // Files should already be opened in HMC_output_single_chain constructor
                                 // But they might have been invalidated by copying...
                                
                                 // Re-open if needed:
                                 if (!HMC_outputs[i].trace_files_is_open()) {
                                    HMC_outputs[i].reopen_trace_files( i, trace_dir);
                                 }
                             
                           }
                    
                           const int chain_id_int = static_cast<int>(i);
                           // const int seed_main_int_i =     global_seed_main_int +     n_iter*(1 + chain_id_int);
                           // const int seed_nuisance_int_i = global_seed_nuisance_int + n_iter*(1 + chain_id_int);
                           // One hashed seed per (run seed, chain, block): the formula above gave chains of different runs the same seed.
                           const int seed_main_int_i =     fn_chain_seed_int(static_cast<std::uint64_t>(global_seed), chain_id_int, 0);
                           const int seed_nuisance_int_i = fn_chain_seed_int(static_cast<std::uint64_t>(global_seed), chain_id_int, 1);
                           
                           #if RNG_TYPE_dqrng_xoshiro256plusplus == 1
                                 dqrng::xoshiro256plus rng_main_i; //(global_rng_main);      // make thread local copy of rng 
                                 dqrng::xoshiro256plus rng_nuisance_i; //(global_rng_nuisance);      // make thread local copy of rng 
                                 rng_main_i.seed(seed_main_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 rng_nuisance_i.seed(seed_nuisance_int_i);  // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           #elif RNG_TYPE_CPP_STD == 1
                                 std::mt19937 rng_main_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 std::mt19937 rng_nuisance_i;  // Fresh RNG // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                                 rng_main_i.seed(seed_main_int_i); // set / re-set the seed
                                 rng_nuisance_i.seed(seed_nuisance_int_i); // set / re-set the seed
                           #endif
                          
                           const int N = Model_args_as_cpp_struct_copies[i].N;
                           const int n_nuisance =  Model_args_as_cpp_struct_copies[i].n_nuisance;
                           const int n_params_main = Model_args_as_cpp_struct_copies[i].n_params_main;
                          
                           // thread_local stan::math::ChainableStack ad_tape; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           stan::math::ChainableStack ad_tape;     // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           //// stan::math::nested_rev_autodiff nested; // bookmark - thread_local works on Linux but not sure about WIndows (also is it needed on Linux?)
                           // pre-touch to force arena growth before sampling
                           // {
                           //   std::vector<double> dummy(N * n_params_main * 2);
                           //   for (auto &d : dummy) d = 1.0;
                           // }
                         
                           const bool burnin_indicator = false;
                           const int current_iter = 0; // gets assigned later for post-burnin
                    
                          chain_start_seconds[i] = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
                          ////
                          //// this chain's state, sized and zero-filled HERE (in parallel) - see the constructor:
                          HMC_inputs[i] = HMCResult(n_params_main, n_nuisance, N);
                          ////
                          {
                    
                              ///////////////////////////////////////// perform iterations for adaptation interval
                              HMC_inputs[i].main_theta_vec() =   theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].main_theta_vec_0() = theta_main_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec() =     theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                              HMC_inputs[i].us_theta_vec_0() =   theta_us_vectors_all_chains_input_from_R_RcppPar.col(i);
                
                              
                              if (Model_type == "Stan") {  
                   
                             
                                           auto t_load_0 = std::chrono::steady_clock::now();
                                           //// (each chain's own model, before the model shared by the chains:)
                                           // Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data(  Model_args_as_cpp_struct_copies[i].model_so_file,
                                           //                                                                             Model_args_as_cpp_struct_copies[i].json_file_path,
                                           //                                                                             seed_main_int_i);
                                           //// (the model shared by the chains: a copy of its handle and
                                           ////  pointers, never destroyed by the chain)
                                           Stan_model_struct  Stan_model_as_cpp_struct =
                                                 Stan_model_shared_by_the_chains;
                                           g_t_load_us += std::chrono::duration_cast<std::chrono::microseconds>(
                                             std::chrono::steady_clock::now() - t_load_0).count();
                                           g_n_load += 1;
                                                                          
                                                                          
                                           // Stan_model_struct  Stan_model_as_cpp_struct = fn_load_Stan_model_and_data( Model_args_as_cpp_struct_copies[i].model_so_file,
                                           //                                                                            Model_args_as_cpp_struct_copies[i].json_file_path,
                                           //                                                                            seed_main_int_i);
                                      
                                           //////////////////////////////// perform iterations for chain i:
                                           //// (a lambda, so that the iterations can run as an untimed block
                                           ////  and then a timed block)
                                           //// (one copy of the metric read by every chain, in place of
                                           ////  EHMC_Metric_as_cpp_struct_copies[i]: the copies are equal and
                                           ////  read-only in sampling, and one copy keeps the nuisance
                                           ////  metric's arrays out of each chain's own cache)
                                           const EHMC_Metric_struct &metric_of_every_chain =
                                                 EHMC_Metric_as_cpp_struct_copies[0];
                                           //// (the flag of the gradient at the chain's current
                                           ////  state, kept from one block of iterations to the
                                           ////  next, so that the first timed iteration reuses the
                                           ////  gradient evaluated before the timed window)
                                           bool gradient_at_the_current_state_is_valid = false;
                                           auto fn_run_iterations = [&](const int n_iterations_of_block) {

                                           fn_sample_HMC_multi_iter_single_thread(   HMC_outputs[i],
                                                                                     HMC_inputs[i], 
                                                                                     burnin_indicator, 
                                                                                     chain_id_int, 
                                                                                     current_iter,
                                                                                     seed_main_int_i, 
                                                                                     seed_nuisance_int_i,
                                                                                     rng_main_i,
                                                                                     rng_nuisance_i,
                                                                                     n_iterations_of_block,
                                                                                     partitioned_HMC,
                                                                                     diffusion_HMC,
                                                                                     Model_type,  sample_nuisance,
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  
                                                                                     n_nuisance_to_track, 
                                                                                     y_copies[i],
                                                                                     // y_copy,
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i],
                                                                                     metric_of_every_chain,
                                                                                     Stan_model_as_cpp_struct, 
                                                                                     LC_MVP_ws_structs_double_vec[i],
                                                                                     // n_threads_WCP);
                                                                                     n_threads_WCP,
                                                                                     &gradient_at_the_current_state_is_valid);

                                           };
                                           //// (the gradient at the chain's current state, as the
                                           ////  joint diffusion sampler evaluates it)
                                           auto fn_lp_grad_at_the_current_state = [&]() {

                                                 fn_lp_grad_InPlace(HMC_inputs[i].lp_and_grad_outs(),
                                                                    Model_type,
                                                                    force_autodiff,
                                                                    force_PartialLog,
                                                                    multi_attempts,
                                                                    HMC_inputs[i].main_theta_vec(),
                                                                    HMC_inputs[i].us_theta_vec(),
                                                                    y_copies[i],
                                                                    "all",
                                                                    Model_args_as_cpp_struct_copies[i],
                                                                    Stan_model_as_cpp_struct,
                                                                    LC_MVP_ws_structs_double_vec[i],
                                                                    n_threads_WCP);

                                           };
                                           //// (timing a layout in one call: the untimed iterations first,
                                           ////  then the start of the timed ones)
                                           const int n_untimed_iterations =
                                                 EHMC_args_as_cpp_struct_copies[i]
                                                       .n_untimed_iterations_before_timed_iterations;
                                           if ((n_untimed_iterations > 0) && (n_untimed_iterations < n_iter)) {
                                                 // fn_run_iterations(n_untimed_iterations);
                                                 fn_run_untimed_iterations_at_their_own_L(
                                                       EHMC_args_as_cpp_struct_copies[i], n_untimed_iterations,
                                                       fn_run_iterations);
                                                 fn_gradient_at_the_starting_state_before_the_timed_iterations(
                                                       gradient_at_the_current_state_is_valid,
                                                       (!partitioned_HMC) && diffusion_HMC && (n_nuisance > 0),
                                                       fn_lp_grad_at_the_current_state,
                                                       HMC_inputs[i].lp_and_grad_outs());
                                                 chain_timed_start_seconds[i] = fn_seconds_on_steady_clock();
                                                 fn_run_iterations(n_iter - n_untimed_iterations);
                                           } else {
                                                 fn_gradient_at_the_starting_state_before_the_timed_iterations(
                                                       gradient_at_the_current_state_is_valid,
                                                       (!partitioned_HMC) && diffusion_HMC && (n_nuisance > 0),
                                                       fn_lp_grad_at_the_current_state,
                                                       HMC_inputs[i].lp_and_grad_outs());
                                                 chain_timed_start_seconds[i] = fn_seconds_on_steady_clock();
                                                 fn_run_iterations(n_iter);
                                           }
                                           chain_timed_end_seconds[i] = fn_seconds_on_steady_clock();
                                            ////////////////////////////// end of iteration(s)
                                            
                                            //// destroy Stan model object:
                                            // fn_bs_destroy_Stan_model_after_sampling(Stan_model_as_cpp_struct);
                                            //// (timing a layout in one call: no release of the memory's pages
                                            ////  here, since handing pages back to the system interrupts every
                                            ////  other chain of the process whilst it is still being timed; the
                                            ////  thread's memory is reused by its next call, as in Stan; the
                                            ////  switch is release_memory_of_the_Stan_model_after_sampling,
                                            ////  separate from the untimed iterations)
                                            // if ((n_untimed_iterations > 0) && (n_untimed_iterations < n_iter)) {
                                            // if (EHMC_args_as_cpp_struct_copies[i]
                                            //           .release_memory_of_the_Stan_model_after_sampling == 0) {
                                            //       fn_bs_destroy_Stan_model(Stan_model_as_cpp_struct);
                                            // } else {
                                            //       fn_bs_destroy_Stan_model_after_sampling(Stan_model_as_cpp_struct);
                                            // }
                                            //// (the model is shared by the chains and destroyed with the
                                            ////  worker: the chain hands back its own thread's memory only)
                                            if (EHMC_args_as_cpp_struct_copies[i]
                                                      .release_memory_of_the_Stan_model_after_sampling != 0) {

                                                  fn_bs_release_this_threads_memory_of_a_shared_Stan_model(
                                                        Stan_model_as_cpp_struct);

                                            }
                        
                              } else { 
                                
                                            Stan_model_struct Stan_model_as_cpp_struct; ///  dummy struct
                                
                                            //////////////////////////////// perform iterations for chain i:
                                            //// (a lambda, so that the iterations can run as an untimed block
                                            ////  and then a timed block)
                                            //// (one copy of the metric read by every chain, as above)
                                            const EHMC_Metric_struct &metric_of_every_chain =
                                                  EHMC_Metric_as_cpp_struct_copies[0];
                                            //// (the flag of the gradient at the chain's current
                                            ////  state, kept from one block of iterations to the
                                            ////  next, so that the first timed iteration reuses the
                                            ////  gradient evaluated before the timed window)
                                            bool gradient_at_the_current_state_is_valid = false;
                                            auto fn_run_iterations = [&](const int n_iterations_of_block) {

                                            fn_sample_HMC_multi_iter_single_thread(  HMC_outputs[i],   
                                                                                     HMC_inputs[i], 
                                                                                     burnin_indicator, 
                                                                                     chain_id_int, 
                                                                                     current_iter,
                                                                                     seed_main_int_i, 
                                                                                     seed_nuisance_int_i,
                                                                                     rng_main_i,
                                                                                     rng_nuisance_i,
                                                                                     n_iterations_of_block,
                                                                                     partitioned_HMC,
                                                                                     diffusion_HMC,
                                                                                     Model_type,  sample_nuisance, 
                                                                                     force_autodiff, force_PartialLog,  multi_attempts,  
                                                                                     n_nuisance_to_track, 
                                                                                     y_copies[i],
                                                                                     // y_copy,
                                                                                     Model_args_as_cpp_struct_copies[i],
                                                                                     EHMC_args_as_cpp_struct_copies[i], 
                                                                                     metric_of_every_chain,
                                                                                     Stan_model_as_cpp_struct, 
                                                                                     LC_MVP_ws_structs_double_vec[i],
                                                                                     // n_threads_WCP);
                                                                                     n_threads_WCP,
                                                                                     &gradient_at_the_current_state_is_valid);

                                            };
                                            //// (the gradient at the chain's current state, as the
                                            ////  joint diffusion sampler evaluates it)
                                            auto fn_lp_grad_at_the_current_state = [&]() {

                                                  fn_lp_grad_InPlace(HMC_inputs[i].lp_and_grad_outs(),
                                                                     Model_type,
                                                                     force_autodiff,
                                                                     force_PartialLog,
                                                                     multi_attempts,
                                                                     HMC_inputs[i].main_theta_vec(),
                                                                     HMC_inputs[i].us_theta_vec(),
                                                                     y_copies[i],
                                                                     "all",
                                                                     Model_args_as_cpp_struct_copies[i],
                                                                     Stan_model_as_cpp_struct,
                                                                     LC_MVP_ws_structs_double_vec[i],
                                                                     n_threads_WCP);

                                            };
                                            //// (timing a layout in one call: the untimed iterations first,
                                            ////  then the start of the timed ones)
                                            const int n_untimed_iterations =
                                                 EHMC_args_as_cpp_struct_copies[i]
                                                       .n_untimed_iterations_before_timed_iterations;
                                            if ((n_untimed_iterations > 0) && (n_untimed_iterations < n_iter)) {
                                                  // fn_run_iterations(n_untimed_iterations);
                                                  fn_run_untimed_iterations_at_their_own_L(
                                                        EHMC_args_as_cpp_struct_copies[i], n_untimed_iterations,
                                                        fn_run_iterations);
                                                  fn_gradient_at_the_starting_state_before_the_timed_iterations(
                                                        gradient_at_the_current_state_is_valid,
                                                        (!partitioned_HMC) && diffusion_HMC && (n_nuisance > 0),
                                                        fn_lp_grad_at_the_current_state,
                                                        HMC_inputs[i].lp_and_grad_outs());
                                                  chain_timed_start_seconds[i] = fn_seconds_on_steady_clock();
                                                  fn_run_iterations(n_iter - n_untimed_iterations);
                                            } else {
                                                  fn_gradient_at_the_starting_state_before_the_timed_iterations(
                                                        gradient_at_the_current_state_is_valid,
                                                        (!partitioned_HMC) && diffusion_HMC && (n_nuisance > 0),
                                                        fn_lp_grad_at_the_current_state,
                                                        HMC_inputs[i].lp_and_grad_outs());
                                                  chain_timed_start_seconds[i] = fn_seconds_on_steady_clock();
                                                  fn_run_iterations(n_iter);
                                            }
                                            chain_timed_end_seconds[i] = fn_seconds_on_steady_clock();
                                            ////////////////////////////// end of iteration(s)
                                            
                              }
                          
                        } // end of parallel stuff
                        ////
                        chain_end_seconds[i] = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
      
          }

    }  //// end of all parallel work// Definition of static thread_local variable
    
    //// debug = TRUE: each iteration's own trajectory length (rows = iterations, columns = chains:
    //// the jittered main-block length tau_main_ii and its leapfrog steps L_main_ii) and each chain's
    //// main-parameter proposals (parameters x iterations). Call it BEFORE move_results_to_output(), which
    //// clears HMC_outputs:
    void copy_debug_record_to_output(Eigen::Matrix<double, -1, -1> &tau_main_ii_output,
                                     Eigen::Matrix<int, -1, -1> &L_main_ii_output,
                                     std::vector<Eigen::Matrix<double, -1, -1>> &proposal_main_output) {
      
            const int n_iter_recorded = HMC_outputs[0].diagnostics_tau_main_ii().size();
            tau_main_ii_output.resize(n_iter_recorded, n_threads);
            L_main_ii_output.resize(n_iter_recorded, n_threads);
            proposal_main_output.resize(n_threads);
            for (int i = 0; i < n_threads; ++i) {
                    tau_main_ii_output.col(i) = HMC_outputs[i].diagnostics_tau_main_ii();
                    L_main_ii_output.col(i) = HMC_outputs[i].diagnostics_L_main_ii();
                    proposal_main_output[i] = HMC_outputs[i].diagnostics_proposal_main();
            }
        
    }
    
    // Copy results directly to R matrices
    void move_results_to_output(Eigen::Matrix<double, -1, -1> &p_jump_main_output,
                                Eigen::Matrix<double, -1, -1> &p_jump_nuisance_output) {
      
            // const int n_nuisance = HMC_outputs[0].trace_nuisance().rows();
            const int n_iter = HMC_outputs[0].trace_main().cols();
            // Preserve the already-recorded acceptance probabilities before clearing HMC_outputs.
            // Rows are iterations; columns are chains. This does not perform any extra sampler work.
            p_jump_main_output.resize(n_iter, n_threads);
            p_jump_nuisance_output.resize(n_iter, sample_nuisance ? n_threads : 0);
      
            trace_output.resize(n_threads);
            trace_output_divs.resize(n_threads);
            
            // // Only resize these if NOT using disk
            if (!use_disk) {
                trace_output_nuisance.resize(n_threads);
                trace_output_log_lik.resize(n_threads);
            }
            
            // const bool use_disk = HMC_outputs[0].using_disk();  // Check if using disk mode (or can pass as a parameter)
            
            for (int i = 0; i < n_threads; ++i) {
                    p_jump_main_output.col(i) = HMC_outputs[i].diagnostics_p_jump_main();
                    if (sample_nuisance) {
                        p_jump_nuisance_output.col(i) = HMC_outputs[i].diagnostics_p_jump_us();
                    }
              
                    // CLOSE FILES FIRST before any swapping/clearing
                    if (use_disk) {
                      HMC_outputs[i].close_trace_files();
                    }
                    
                    // Main and div - always swap (always in RAM)
                    trace_output[i].swap(HMC_outputs[i].trace_main());
                    trace_output_divs[i].swap(HMC_outputs[i].trace_div());
                    
                    // Nuisance and log_lik - only swap if NOT using disk
                    if (use_disk == false) {
                      if (sample_nuisance == true) {
                        trace_output_nuisance[i].swap(HMC_outputs[i].trace_nuisance());
                      }
                      if (Model_type != "Stan") {
                        trace_output_log_lik[i].swap(HMC_outputs[i].trace_log_lik());
                      }
                    }
                    
                    // FREE THIS CHAIN'S MEMORY IMMEDIATELY AFTER COPYING
                    HMC_outputs[i].trace_main().resize(0, 0);
                    HMC_outputs[i].trace_div().resize(0, 0);
                    HMC_outputs[i].trace_nuisance().resize(0, 0);
                    HMC_outputs[i].trace_log_lik().resize(0, 0);
                    
            }
            
            // Clear the vectors themselves
            HMC_outputs.clear();
            HMC_outputs.shrink_to_fit();
            HMC_inputs.clear();
            HMC_inputs.shrink_to_fit();
        
    }
  
    std::string get_trace_dir() const {

      if (HMC_outputs.size() > 0 && HMC_outputs[0].using_disk()) {
        return HMC_outputs[0].trace_dir();
      }
      return "";

    }
    
    bool is_using_disk() const {

      return HMC_outputs.size() > 0 && HMC_outputs[0].using_disk();

    }

};



 
 
 
 





  









