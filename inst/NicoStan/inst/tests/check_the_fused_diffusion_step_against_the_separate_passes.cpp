// Check of the fused nuisance step of
// leapfrog_integrator_dense_M_diffusion_HMC_dual_Id_M_us_zero_theta_us_InPlace (EHMC_dual_sampler_fns.hpp):
// L steps of the separate Eigen passes it replaced (copied verbatim) against L steps of the one-pass loop, on
// the same start and the same gradient, compared element by element. The gradient stands in for the model's:
// g(theta) = -theta + 0.3 sin(theta) (any smooth function of theta; the step only reads it).
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

static void gradient(const Eigen::VectorXd& theta, Eigen::VectorXd& g) {
    g = -theta + 0.3 * theta.array().sin().matrix();
}

int main() {
    for (int n : {1, 3, 4, 7, 8, 9, 31, 257, 5000, 25000}) {
        for (int L : {1, 2, 10}) {
            std::mt19937 generator(123 + n + L);
            std::normal_distribution<double> normal(0.0, 1.0);
            Eigen::VectorXd theta_start(n), velocity_start(n);
            for (int i = 0; i < n; ++i) {
                theta_start[i] = normal(generator);
                velocity_start[i] = normal(generator);
            }
            const double eps = 0.37;
            const double cos_eps = std::cos(eps), sin_eps = std::sin(eps), half_eps = 0.5 * eps;
            //// ---- the separate passes (as in the source before the fusion)
            Eigen::VectorXd theta_a = theta_start, velocity_a = velocity_start, g_a(n);
            Eigen::VectorXd seg_theta(n), seg_velocity(n);
            gradient(theta_a, g_a);
            for (int l = 0; l < L; ++l) {
                velocity_a.array() += 0.5 * eps * (g_a.array() + theta_a.array());
                seg_theta = theta_a;
                seg_velocity = velocity_a;
                theta_a.array() = cos_eps * seg_theta.array() + sin_eps * seg_velocity.array();
                velocity_a.array() = cos_eps * seg_velocity.array() - sin_eps * seg_theta.array();
                gradient(theta_a, g_a);
                velocity_a.array() += 0.5 * eps * (g_a.array() + theta_a.array());
            }
            //// ---- the one-pass loop (as in the source after the fusion)
            Eigen::VectorXd theta_b = theta_start, velocity_b = velocity_start, g_b(n);
            gradient(theta_b, g_b);
            double* __restrict v = velocity_b.data();
            double* __restrict t = theta_b.data();
            const double* __restrict g = g_b.data();
            for (int l = 0; l < L; ++l) {
                if (l == 0) {
                    for (int i = 0; i < n; ++i) {
                        const double kick = half_eps * (g[i] + t[i]);
                        const double velocity = v[i] + kick;
                        const double th = t[i];
                        t[i] = cos_eps * th + sin_eps * velocity;
                        v[i] = cos_eps * velocity - sin_eps * th;
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        const double kick = half_eps * (g[i] + t[i]);
                        const double velocity = (v[i] + kick) + kick;
                        const double th = t[i];
                        t[i] = cos_eps * th + sin_eps * velocity;
                        v[i] = cos_eps * velocity - sin_eps * th;
                    }
                }
                gradient(theta_b, g_b);
            }
            for (int i = 0; i < n; ++i) v[i] += half_eps * (g[i] + t[i]);
            const double largest_difference = std::max((theta_a - theta_b).cwiseAbs().maxCoeff(),
                                                       (velocity_a - velocity_b).cwiseAbs().maxCoeff());
            std::printf("n = %6d, L = %2d: largest difference %.3e%s\n", n, L, largest_difference,
                        largest_difference == 0.0 ? " (identical)" : "");
            if (largest_difference > 1e-12) return 1;
        }
    }
    //// ---- the general-metric step (leapfrog_integrator_dense_M_ / diag_M_diffusion_HMC_dual_InPlace): a mass
    ////      matrix M and a centre theta_hat of their own for every nuisance parameter
    for (int n : {1, 3, 4, 7, 8, 9, 31, 257, 5000, 25000}) {
        for (int L : {1, 2, 10}) {
            std::mt19937 generator(321 + n + L);
            std::normal_distribution<double> normal(0.0, 1.0);
            Eigen::VectorXd theta_start(n), velocity_start(n), M(n), theta_hat(n);
            for (int i = 0; i < n; ++i) {
                theta_start[i] = normal(generator);
                velocity_start[i] = normal(generator);
                M[i] = std::exp(0.5 * normal(generator));
                theta_hat[i] = 0.3 * normal(generator);
            }
            const Eigen::VectorXd M_inv = M.cwiseInverse();
            const double eps = 0.37;
            const Eigen::VectorXd sqrt_M = M.array().sqrt();
            const Eigen::VectorXd inv_sqrt_M = M_inv.array().sqrt();
            const Eigen::VectorXd angle = eps * inv_sqrt_M;
            const Eigen::VectorXd cos_t = angle.array().cos();
            const Eigen::VectorXd sin_t = angle.array().sin();
            //// ---- the separate passes (as in the source before the fusion)
            Eigen::VectorXd theta_a = theta_start, velocity_a = velocity_start, g_a(n);
            Eigen::VectorXd seg_theta(n), seg_velocity(n);
            gradient(theta_a, g_a);
            for (int l = 0; l < L; ++l) {
                velocity_a.array() += 0.5 * eps * M_inv.array() * (g_a.array() + theta_a.array()
                                                                    - theta_hat.array());
                seg_theta.array() = theta_a.array() - theta_hat.array();
                seg_velocity = velocity_a;
                theta_a.array() = cos_t.array() * seg_theta.array()
                                  + sqrt_M.array() * sin_t.array() * seg_velocity.array() + theta_hat.array();
                velocity_a.array() = cos_t.array() * seg_velocity.array()
                                     - inv_sqrt_M.array() * sin_t.array() * seg_theta.array();
                gradient(theta_a, g_a);
                velocity_a.array() += 0.5 * eps * M_inv.array() * (g_a.array() + theta_a.array()
                                                                    - theta_hat.array());
            }
            //// ---- the one-pass loop (as in the source after the fusion)
            const Eigen::VectorXd sqrt_M_sin_t = (sqrt_M.array() * sin_t.array()).matrix();
            const Eigen::VectorXd inv_sqrt_M_sin_t = (inv_sqrt_M.array() * sin_t.array()).matrix();
            const double half_eps = 0.5 * eps;
            Eigen::VectorXd theta_b = theta_start, velocity_b = velocity_start, g_b(n);
            gradient(theta_b, g_b);
            double* __restrict v = velocity_b.data();
            double* __restrict t = theta_b.data();
            const double* __restrict g = g_b.data();
            for (int l = 0; l < L; ++l) {
                for (int i = 0; i < n; ++i) {
                    const double kick = half_eps * M_inv[i] * ((g[i] + t[i]) - theta_hat[i]);
                    const double velocity = l > 0 ? (v[i] + kick) + kick : v[i] + kick;
                    const double distance = t[i] - theta_hat[i];
                    t[i] = cos_t[i] * distance + sqrt_M_sin_t[i] * velocity + theta_hat[i];
                    v[i] = cos_t[i] * velocity - inv_sqrt_M_sin_t[i] * distance;
                }
                gradient(theta_b, g_b);
            }
            for (int i = 0; i < n; ++i) v[i] += half_eps * M_inv[i] * ((g[i] + t[i]) - theta_hat[i]);
            const double largest_difference = std::max((theta_a - theta_b).cwiseAbs().maxCoeff(),
                                                       (velocity_a - velocity_b).cwiseAbs().maxCoeff());
            std::printf("general metric: n = %6d, L = %2d: largest difference %.3e\n", n, L, largest_difference);
            if (largest_difference > 1e-12) return 1;
        }
    }
    std::printf("fused diffusion step check passed\n");
    return 0;
}
