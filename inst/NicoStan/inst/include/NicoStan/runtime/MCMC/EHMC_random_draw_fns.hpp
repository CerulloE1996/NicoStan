
#pragma once

 


#include <random>
#include <cstdint>

  

#include <Eigen/Dense>
 
 
 

#include <unsupported/Eigen/SpecialFunctions>
 
 
 
 
using namespace Eigen;
 
 
 
 
 
 


 

 

 
// HMC sampler functions   ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Burn-in calls the single-thread loop with one iteration at a time. Its incoming seed
// already includes iteration + chain, so adding them again cannot distinguish pairs such
// as (iteration 2, chain 1) and (iteration 1, chain 2). Seed from separate tuple fields.
template<typename T>
inline void fn_seed_burnin_rng(T &rng,
                               const int incoming_chain_seed,
                               const int chain_id,
                               const int iteration,
                               const std::uint32_t parameter_block) {
    std::seed_seq seed_sequence{static_cast<std::uint32_t>(incoming_chain_seed),
                               static_cast<std::uint32_t>(chain_id),
                               static_cast<std::uint32_t>(iteration),
                               parameter_block, std::uint32_t{0x4255524e}};
#if RNG_TYPE_CPP_STD == 1
    rng.seed(seed_sequence);
#elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
    std::uint32_t seed_words[2];
    seed_sequence.generate(seed_words, seed_words + 2);
    const std::uint64_t combined_seed = (static_cast<std::uint64_t>(seed_words[0]) << 32) | seed_words[1];
    rng.seed(combined_seed);
#endif
}


// Chain seeds: one hashed seed per (run seed, chain, parameter block). The earlier formula
// seed + n_iter*(1 + chain) gave the same integer to chains of different runs (e.g. run seed 1000
// chain 4 and run seed 2000 chain 0 at n_iter = 250), so different runs shared random streams.
inline int fn_chain_seed_int(const std::uint64_t global_seed,
                             const int chain_id,
                             const std::uint32_t parameter_block) {
    std::seed_seq seed_sequence{static_cast<std::uint32_t>(global_seed & 0xffffffffULL),
                               static_cast<std::uint32_t>(global_seed >> 32),
                               static_cast<std::uint32_t>(chain_id),
                               parameter_block, std::uint32_t{0x43484e53}};
    std::uint32_t seed_word[1];
    seed_sequence.generate(seed_word, seed_word + 1);
    return static_cast<int>(seed_word[0] & 0x7fffffffU);
}

// Sampling iterations: seed from separate tuple fields (chain seed, chain, iteration, block), as the
// burn-in does, with a different tag so that sampling and burn-in streams never coincide.
template<typename T>
inline void fn_seed_sampling_rng(T &rng,
                                 const int incoming_chain_seed,
                                 const int chain_id,
                                 const int iteration,
                                 const std::uint32_t parameter_block) {
    std::seed_seq seed_sequence{static_cast<std::uint32_t>(incoming_chain_seed),
                               static_cast<std::uint32_t>(chain_id),
                               static_cast<std::uint32_t>(iteration),
                               parameter_block, std::uint32_t{0x53414d50}};
#if RNG_TYPE_CPP_STD == 1
    rng.seed(seed_sequence);
#elif RNG_TYPE_dqrng_xoshiro256plusplus == 1
    std::uint32_t seed_words[2];
    seed_sequence.generate(seed_words, seed_words + 2);
    const std::uint64_t combined_seed = (static_cast<std::uint64_t>(seed_words[0]) << 32) | seed_words[1];
    rng.seed(combined_seed);
#endif
}

 
 


template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE   Eigen::Matrix<double, -1, 1>  generate_random_std_norm_vec( int n_params, 
                                                                            T &rng) {
   
         //// Initialise at zero:
         Eigen::Matrix<double, -1, 1> std_norm_vec = Eigen::Matrix<double, -1, 1>::Zero(n_params);
         
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw;
         }
         
         return std_norm_vec;
   
}
 
 
 
 
 
 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE    void generate_random_std_norm_vec_InPlace( Eigen::Ref<Eigen::Matrix<double, -1, 1>> std_norm_vec,
                                                            T &rng) {
  
         const int n_params = std_norm_vec.size();
    
         std::normal_distribution<double> dist(0.0, 1.0); 
         
         //// Initialise at zero:
         std_norm_vec.setZero();
         
         //// Fill vector:
         for (int d = 0; d < n_params; d++) {
             double norm_draw = dist(rng);
             std_norm_vec(d) = norm_draw; 
         }
   
}
 
 
 
  
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_std_uniform(T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double rand_std_unif = dist(rng);
        return rand_std_unif;
  
}


 
template<typename T = RNG_TYPE_dqrng>
ALWAYS_INLINE  double generate_random_tau_ii(  double tau, 
                                               T &rng) {
  
        std::uniform_real_distribution<double> dist(0.0, 2.0 * tau);
        const double tau_ii = dist(rng);
        return tau_ii;

}


//// The override belongs to one burn-in worker call; it never changes the shared sampler argument layout.
struct BurninTauJitterOverride {
    bool active = false;
    double tau_main_ii = 0.0;
    double tau_us_ii = 0.0;
};

inline BurninTauJitterOverride &fn_burnin_tau_jitter_override() {
    static thread_local BurninTauJitterOverride state;
    return state;
}

inline void fn_clear_burnin_tau_jitter_override() {
    fn_burnin_tau_jitter_override() = BurninTauJitterOverride{};
}

class BurninTauJitterScope {
    const BurninTauJitterOverride previous_state;

public:
    BurninTauJitterScope(const bool active, const double tau_main_ii, const double tau_us_ii)
      : previous_state(fn_burnin_tau_jitter_override()) {
        fn_burnin_tau_jitter_override() = BurninTauJitterOverride{active, tau_main_ii, tau_us_ii};
    }

    ~BurninTauJitterScope() {
        fn_burnin_tau_jitter_override() = previous_state;
    }

    BurninTauJitterScope(const BurninTauJitterScope &) = delete;
    BurninTauJitterScope &operator=(const BurninTauJitterScope &) = delete;
};

template<typename T>
ALWAYS_INLINE double fn_generate_burnin_tau_ii(const double tau, T &rng, const bool main_block) {
    const BurninTauJitterOverride &state = fn_burnin_tau_jitter_override();
    if (state.active) return main_block ? state.tau_main_ii : state.tau_us_ii;
    return generate_random_tau_ii(tau, rng);
}





 
 
 

 
 
 
 
 
 
 
 
 
 

 
 
