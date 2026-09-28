#pragma once

#include <stan/math/rev/core/chainablestack.hpp>
#include <stan/math/rev/core/recover_memory.hpp>

// Include in the compiled model, whose hidden TLS stack differs from NicoStan's.
// The caller must use the same worker after its complete sampling call returns.
#ifdef _WIN32
#define NICOSTAN_BRIDGE_MEMORY_EXPORT __declspec(dllexport)
#else
#define NICOSTAN_BRIDGE_MEMORY_EXPORT __attribute__((visibility("default")))
#endif

// 0: released/already empty; 1: live AD state, untouched; 2: unexpected failure.
extern "C" NICOSTAN_BRIDGE_MEMORY_EXPORT
int nicostan_bs_release_thread_memory_v1() noexcept {
    try {
        auto* stack = stan::math::ChainableStack::instance_;
        if (stack == nullptr) return 0;
        if (!stack->var_stack_.empty() || !stack->var_nochain_stack_.empty()
            || !stack->var_alloc_stack_.empty()
            || !stack->nested_var_stack_sizes_.empty()
            || !stack->nested_var_nochain_stack_sizes_.empty()
            || !stack->nested_var_alloc_stack_starts_.empty()) return 1;

        // Gradient's nested RAII scope has already recovered all its variables.
        // Rewind only after proving there are no live variables or nested scopes.
        stan::math::recover_memory();
#ifdef __linux__
        stack->memalloc_.nicostan_discard_unused_pages();
        stan::math::stack_alloc::nicostan_discard_pages(
            stack->var_stack_.data(), stack->var_stack_.capacity()
            * sizeof(decltype(stack->var_stack_)::value_type));
        stan::math::stack_alloc::nicostan_discard_pages(
            stack->var_nochain_stack_.data(), stack->var_nochain_stack_.capacity()
            * sizeof(decltype(stack->var_nochain_stack_)::value_type));
        stan::math::stack_alloc::nicostan_discard_pages(
            stack->var_alloc_stack_.data(), stack->var_alloc_stack_.capacity()
            * sizeof(decltype(stack->var_alloc_stack_)::value_type));
#endif
        return 0;
    } catch (...) {
        return 2;
    }
}

#undef NICOSTAN_BRIDGE_MEMORY_EXPORT






















