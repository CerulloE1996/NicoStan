#pragma once
// Optional extension: real whole-vector autodiff and ordinary normal-interval fusion.
// The existing header, mathematical kernels and AoS overloads remain unchanged.
#include "BayesMVP_AVX_stan_fns.hpp"

namespace bmvp_stan_ext {

// A single vector-valued node owns contiguous values AND adjoints. Keep the caller in this
// representation across operations; converting back to Matrix<var> at every call defeats it.
template <derivative_values needed, typename T, typename V, typename D,
          typename K = std::nullptr_t>
inline auto varmat_apply(const T& input, V value, D derivative, K kernel = nullptr) {
    const int n = input.size();
    Eigen::VectorXd output(n);
    if constexpr (std::is_same<K, std::nullptr_t>::value) {
        const Eigen::VectorXd argument = input.val();
        value(argument.data(), n, output.data());
    } else {
        output = input.val();
        kernel(output.data(), n);
    }
    return stan::math::make_callback_var(std::move(output), [input, derivative, n](auto& result) mutable {
        scratch_t& buffers = scratch(); buffers.ensure(n);
        const double* x = nullptr;
        if constexpr (needed != derivative_values::result) {
            // Plain vector-valued vars are contiguous. Also support strided column views safely.
            if (input.val().innerStride() == 1) x = input.val().data();
            else {
                for (int i = 0; i < n; ++i) buffers.a[i] = input.val().coeff(i);
                x = buffers.a.data();
            }
        }
        const double* y = needed == derivative_values::operand ? nullptr : result.val_.data();
        derivative(x, y, n, buffers.c.data());
        input.adj().array() += result.adj_.array()
            * Eigen::Map<const Eigen::VectorXd>(buffers.c.data(), n).array();
    });
}
} // namespace bmvp_stan_ext

#define BMVP_VARMAT(NAME, NEEDED, KERNEL)                                                        \
template <typename T, stan::require_var_col_vector_t<T>* = nullptr>                             \
inline auto fast_##NAME(const T& input, std::ostream* /*pstream__*/) {                           \
    return bmvp_stan_ext::varmat_apply<bmvp_stan_ext::derivative_values::NEEDED>(input,           \
        [](const double* x, int n, double* y) { bmvp_stan_ext::v_##NAME(x, n, y); },              \
        [](const double* x, const double* y, int n, double* d) { bmvp_stan_ext::d_##NAME(x,y,n,d); }, KERNEL); \
}
#define BMVP_VECTOR_KERNEL(NAME) [](double* x, int n) { bmvp_stan_ext::k_##NAME(x,n); }
BMVP_VARMAT(Phi, operand, BMVP_VECTOR_KERNEL(Phi))
BMVP_VARMAT(log_Phi, operand, BMVP_VECTOR_KERNEL(log_Phi))
BMVP_VARMAT(inv_Phi, result, BMVP_VECTOR_KERNEL(inv_Phi))
BMVP_VARMAT(exp, result, BMVP_VECTOR_KERNEL(exp))
BMVP_VARMAT(log, operand, BMVP_VECTOR_KERNEL(log))
BMVP_VARMAT(log1m, operand, BMVP_VECTOR_KERNEL(log1m))
BMVP_VARMAT(log1p, operand, BMVP_VECTOR_KERNEL(log1p))
BMVP_VARMAT(log1p_exp, operand, BMVP_VECTOR_KERNEL(log1p_exp))
BMVP_VARMAT(log1m_exp, both, nullptr)
BMVP_VARMAT(inv_logit, result, BMVP_VECTOR_KERNEL(inv_logit))
BMVP_VARMAT(log_inv_logit, operand, BMVP_VECTOR_KERNEL(log_inv_logit))
BMVP_VARMAT(Phi_approx, both, BMVP_VECTOR_KERNEL(Phi_approx))
BMVP_VARMAT(log_Phi_approx, both, BMVP_VECTOR_KERNEL(log_Phi_approx))
BMVP_VARMAT(inv_Phi_approx, both, BMVP_VECTOR_KERNEL(inv_Phi_approx))
BMVP_VARMAT(inv_Phi_approx_from_logit_prob, both, BMVP_VECTOR_KERNEL(inv_Phi_approx_from_logit_prob))
#undef BMVP_VARMAT
#undef BMVP_VECTOR_KERNEL

namespace bmvp_stan_ext {

template <typename T> constexpr bool differentiable_operand =
    stan::is_var_matrix<T>::value || stan::is_var<stan::scalar_type_t<T>>::value;

template <typename T>
inline double operand_value(const T& x, int i) {
    if constexpr (stan::is_var_matrix<T>::value) return x.val().coeff(i);
    else return stan::math::value_of(x.coeff(i));
}
template <typename T>
inline void add_operand_adjoint(T& x, int i, double value) {
    if constexpr (stan::is_var_matrix<T>::value) x.adj().coeffRef(i) += value;
    else if constexpr (stan::is_var<stan::scalar_type_t<T>>::value) x.coeff(i).adj() += value;
}

// Return N x 2: column 1 is z, column 2 log(probability). One callback; never fake
// scalar nodes or omit their normal registration/reset. AoS scatter is ordered for aliases.
template <bool vector_result, typename Reverse>
inline auto pack_fused_result(Eigen::MatrixXd values, Reverse reverse) {
    if constexpr (vector_result) {
        return stan::math::make_callback_var(std::move(values), [reverse](auto& result) mutable {
            reverse([&](int i, int col) { return result.val_(i,col); },
                    [&](int i, int col) { return result.adj_(i,col); });
        });
    } else {
        const int n = values.rows();
        vari* nodes = stan::math::ChainableStack::instance_->memalloc_.alloc_array<vari>(2*n);
        Eigen::Matrix<var, -1, -1> output(n,2);
        for (int j=0; j<2*n; ++j) {
            ::new (static_cast<void*>(nodes+j)) vari(values.data()[j],false);
            output.data()[j] = var(nodes+j);
        }
        stan::math::reverse_pass_callback([nodes,n,reverse]() mutable {
            reverse([&](int i, int col) { return nodes[i+col*n].val_; },
                    [&](int i, int col) { return nodes[i+col*n].adj_; });
        });
        return output;
    }
}

// Ordinary interval only. The Stan caller retains its existing selection of stable
// tails/narrow intervals/Phi_type=2. Do not clamp probabilities or change that selection.
template <typename A, typename B, typename U>
inline auto normal_interval(const A& lower, const B& upper, const U& uniform) {
    const int n = lower.size();
    if (upper.size()!=n || uniform.size()!=n) throw std::invalid_argument("fast_normal_interval: vector lengths differ");
    Eigen::VectorXd p_lower(n), p_upper(n), mass(n);
    for (int i=0; i<n; ++i) { p_lower[i]=operand_value(lower,i); p_upper[i]=operand_value(upper,i); }
    k_Phi(p_lower.data(),n); k_Phi(p_upper.data(),n);
    Eigen::MatrixXd output(n,2);
    for (int i=0; i<n; ++i) {
        mass[i]=p_upper[i]-p_lower[i];
        output(i,0)=p_lower[i]+mass[i]*operand_value(uniform,i);
        output(i,1)=mass[i];
    }
    k_inv_Phi(output.col(0).data(),n); k_log(output.col(1).data(),n);
    if constexpr (!(differentiable_operand<A> || differentiable_operand<B> || differentiable_operand<U>)) return output;
    else {
        auto a=stan::math::to_arena(lower), b=stan::math::to_arena(upper), u=stan::math::to_arena(uniform);
        stan::math::arena_matrix<Eigen::VectorXd> probability=mass;
        auto reverse=[a,b,u,probability,n](auto val,auto adj) mutable {
            for (int first=0; first<n; first+=BMVP_LANES) {
                const int count=std::min(BMVP_LANES,n-first);
                alignas(64) double av[BMVP_LANES],bv[BMVP_LANES],z[BMVP_LANES];
                alignas(64) double da[BMVP_LANES],db[BMVP_LANES],dz[BMVP_LANES];
                for(int j=0;j<count;++j) { av[j]=operand_value(a,first+j); bv[j]=operand_value(b,first+j); z[j]=val(first+j,0); }
                d_Phi(av,nullptr,count,da); d_Phi(bv,nullptr,count,db); d_inv_Phi(nullptr,z,count,dz);
                for(int j=0;j<count;++j) {
                    const int i=first+j;
                    const double q=adj(i,0)*dz[j], r=adj(i,1)/probability[i], uv=operand_value(u,i);
                    add_operand_adjoint(a,i,(q*(1.0-uv)-r)*da[j]);
                    add_operand_adjoint(b,i,(q*uv+r)*db[j]);
                    add_operand_adjoint(u,i,q*probability[i]);
                }
            }
        };
        return pack_fused_result<stan::is_var_matrix<A>::value || stan::is_var_matrix<B>::value || stan::is_var_matrix<U>::value>(std::move(output),reverse);
    }
}

template <typename T> inline auto scalar_node_view(const T& input) {
    if constexpr(stan::is_var_matrix<T>::value) return stan::math::from_var_value(input);
    else return stan::math::eval(input);
}

// At u extremely close to 0/1, the original reverse graph has cancellation of
// very large terms. Preserve that graph at these exceptional inputs; do not claim
// floating-point equivalence of an algebraically regrouped derivative there.
template <typename B,typename U,typename Y>
inline auto normal_binary_unfused(const B& bound,const U& uniform,const Y& outcome) {
    auto b=scalar_node_view(bound),u=scalar_node_view(uniform);
    Eigen::VectorXd y(outcome.size());
    for(int i=0;i<y.size();++i)y[i]=operand_value(outcome,i);
    Eigen::VectorXd s=(2*y.array()-1).matrix();
    auto p=fast_Phi(b,nullptr);
    auto r=stan::math::add(stan::math::elt_multiply(y,p),stan::math::elt_multiply(stan::math::elt_multiply(stan::math::subtract(y,p),s),u));
    auto probability=stan::math::add(stan::math::elt_multiply(y,stan::math::subtract(1.0,p)),
        stan::math::elt_multiply(stan::math::elt_multiply(stan::math::subtract(y,1.0),p),s));
    auto z=fast_inv_Phi(r,nullptr),lp=fast_log(probability,nullptr);
    if constexpr(stan::is_var_matrix<B>::value || stan::is_var_matrix<U>::value)
        return stan::math::to_var_value(stan::math::append_col(z,lp));
    else return stan::math::append_col(z,lp).eval();
}

template <typename B, typename U, typename Y>
inline auto normal_binary(const B& bound, const U& uniform, const Y& outcome) {
    // outcome is a discrete 0/1 mask, never differentiated. stanc may promote a
    // locally indexed data vector to Matrix<var>; its values still define the mask.
    const int n=bound.size();
    if (uniform.size()!=n || outcome.size()!=n) throw std::invalid_argument("fast_normal_binary: vector lengths differ");
    Eigen::VectorXd cdf(n),mass(n);
    bool retain_unfused=false;
    for(int i=0;i<n;++i) {
        cdf[i]=operand_value(bound,i);
        const double y=operand_value(outcome,i);
        if(y!=0.0 && y!=1.0) throw std::domain_error("fast_normal_binary: outcomes must be 0 or 1");
        const double uv=operand_value(uniform,i);
        retain_unfused=retain_unfused || uv<1e-12 || uv>1.0-1e-12;
    }
    if(retain_unfused) return normal_binary_unfused(bound,uniform,outcome);
    k_Phi(cdf.data(),n);
    Eigen::MatrixXd output(n,2);
    for(int i=0;i<n;++i) {
        const double p=cdf[i], y=operand_value(outcome,i), s=y+(y-1.0);
        // Preserve the existing model's expression/order rather than simplify away numerical terms.
        mass[i]=y*(1.0-p)+(y-1.0)*p*s;
        output(i,0)=y*p+(y-p)*s*operand_value(uniform,i);
        output(i,1)=mass[i];
    }
    k_inv_Phi(output.col(0).data(),n); k_log(output.col(1).data(),n);
    if constexpr (!(differentiable_operand<B> || differentiable_operand<U>)) return output;
    else {
        auto b=stan::math::to_arena(bound),u=stan::math::to_arena(uniform),y=stan::math::to_arena(outcome);
        stan::math::arena_matrix<Eigen::VectorXd> p=cdf, probability=mass;
        auto reverse=[b,u,y,p,probability,n](auto val,auto adj) mutable {
            for(int first=0;first<n;first+=BMVP_LANES) {
                const int count=std::min(BMVP_LANES,n-first);
                alignas(64) double bv[BMVP_LANES],z[BMVP_LANES],db[BMVP_LANES],dz[BMVP_LANES];
                for(int j=0;j<count;++j) { bv[j]=operand_value(b,first+j); z[j]=val(first+j,0); }
                d_Phi(bv,nullptr,count,db); d_inv_Phi(nullptr,z,count,dz);
                for(int j=0;j<count;++j) {
                    const int i=first+j;
                    const double yy=operand_value(y,i),s=yy+(yy-1.0),uv=operand_value(u,i);
                    const double q=adj(i,0)*dz[j],r=adj(i,1)/probability[i];
                    add_operand_adjoint(b,i,(q*(yy-s*uv)+r*(-yy+(yy-1.0)*s))*db[j]);
                    add_operand_adjoint(u,i,q*(yy-p[i])*s);
                }
            }
        };
        return pack_fused_result<stan::is_var_matrix<B>::value || stan::is_var_matrix<U>::value>(std::move(output),reverse);
    }
}
} // namespace bmvp_stan_ext

template <typename A, typename B, typename U>
inline auto fast_normal_interval(const A& lower,const B& upper,const U& uniform,std::ostream*) {
    return bmvp_stan_ext::normal_interval(lower,upper,uniform);
}
template <typename B, typename U, typename Y>
inline auto fast_normal_binary(const B& bound,const U& uniform,const Y& outcome,std::ostream*) {
    return bmvp_stan_ext::normal_binary(bound,uniform,outcome);
}
