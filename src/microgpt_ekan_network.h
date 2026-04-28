/*
 * microgpt_ekan_network.h — C99 EKAN Autoencoder Network
 *
 * Full Entangled Kolmogorov-Arnold Network for manifold learning.
 * Ported from EnX-cpp ekan_engine.hpp (C++17).
 *
 * Architecture: Multi-layer network with:
 *   - Fourier basis activations (2N+1 coefficients per edge)
 *   - Entanglement mixing weights (W_ji × x_i per connection)
 *   - Dual activation: SiLU(z) + Fourier(z)
 *   - Residual gating: σ(α) × kan + (1-σ(α)) × skip
 *   - Adam optimizer with gradient clipping
 *
 * Default topology: {12, 8, 4, 8, 12} for fraud autoencoder
 *
 * Copyright (c) 2026 Ajay Soni (ajay.soni@enjector.com), Enjector Software Ltd.
 * MIT License — see LICENSE file for details.
 */

#ifndef MICROGPT_EKAN_NETWORK_H
#define MICROGPT_EKAN_NETWORK_H

#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* =========================================================================
 * Configuration
 * ========================================================================= */
#define EKAN_NET_MAX_LAYERS   5     /* max layers (edges between nodes) */
#define EKAN_NET_MAX_DIM     16     /* max nodes per layer */
#define EKAN_NET_FOURIER_N    5     /* frequency terms → 11 coeffs/edge */
#define EKAN_NET_FOURIER_COEFFS (2 * EKAN_NET_FOURIER_N + 1)  /* = 11 */

/* Adam optimizer defaults */
#define EKAN_ADAM_BETA1  0.9
#define EKAN_ADAM_BETA2  0.999
#define EKAN_ADAM_EPS    1e-8

/* Training defaults */
#define EKAN_MAX_GRAD_NORM  1.0
#define EKAN_L2_LAMBDA      1e-4

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* =========================================================================
 * Fourier Basis (per edge)
 * ========================================================================= */

/* Evaluate: f(x) = a₀ + Σ [aₙcos(nπx) + bₙsin(nπx)] */
static inline double ekan_fourier_eval(const double *coeffs, int n_freqs, double x) {
    double result = coeffs[0]; /* a₀ (DC) */
    for (int n = 1; n <= n_freqs; n++) {
        double angle = n * M_PI * x;
        result += coeffs[2*n - 1] * cos(angle);  /* aₙ */
        result += coeffs[2*n]     * sin(angle);   /* bₙ */
    }
    return result;
}

/* Derivative: f'(x) = Σ nπ[-aₙsin(nπx) + bₙcos(nπx)] */
static inline double ekan_fourier_deriv(const double *coeffs, int n_freqs, double x) {
    double result = 0.0;
    for (int n = 1; n <= n_freqs; n++) {
        double angle = n * M_PI * x;
        double freq = n * M_PI;
        result += freq * (-coeffs[2*n - 1] * sin(angle) + coeffs[2*n] * cos(angle));
    }
    return result;
}

/* Basis values at x: [1, cos(πx), sin(πx), cos(2πx), sin(2πx), ...] */
static inline void ekan_fourier_basis(int n_freqs, double x, double *basis_out) {
    basis_out[0] = 1.0;
    for (int n = 1; n <= n_freqs; n++) {
        double angle = n * M_PI * x;
        basis_out[2*n - 1] = cos(angle);
        basis_out[2*n]     = sin(angle);
    }
}

/* =========================================================================
 * SiLU activation: x × σ(x) = x / (1 + e^(-x))
 * ========================================================================= */

static inline double ekan_silu(double x) {
    return x / (1.0 + exp(-x));
}

/* d/dx[SiLU] = σ(x) + x·σ(x)·(1-σ(x)) = (1 + e^(-x) + x·e^(-x)) / (1+e^(-x))² */
static inline double ekan_silu_deriv(double x) {
    double ex = exp(-x);
    double d = 1.0 + ex;
    return (1.0 + ex + x * ex) / (d * d);
}

/* =========================================================================
 * Gradient clipping
 * ========================================================================= */

static inline double ekan_clip_grad(double g, double max_norm) {
    if (g > max_norm)  return max_norm;
    if (g < -max_norm) return -max_norm;
    return g;
}

/* =========================================================================
 * EKAN Layer
 * ========================================================================= */

typedef struct {
    int in_dim;
    int out_dim;

    /* Entanglement mixing weights: W[j][i] */
    double mixing[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM];

    /* Fourier coefficients per edge: fourier[j][i][c] */
    double fourier[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM][EKAN_NET_FOURIER_COEFFS];

    /* Residual gate α */
    double gate_alpha;

    /* Skip projection weights (when in_dim != out_dim) */
    double skip_weights[EKAN_NET_MAX_DIM * EKAN_NET_MAX_DIM];

    /* Cached input from last forward pass (for backward) */
    double last_input[EKAN_NET_MAX_DIM];

    /* Adam moments for mixing weights */
    double m_mixing[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM];
    double v_mixing[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM];

    /* Adam moments for Fourier coefficients */
    double m_fourier[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM][EKAN_NET_FOURIER_COEFFS];
    double v_fourier[EKAN_NET_MAX_DIM][EKAN_NET_MAX_DIM][EKAN_NET_FOURIER_COEFFS];

    /* Adam moments for gate */
    double m_gate, v_gate;

    /* Adam moments for skip projection */
    double m_skip[EKAN_NET_MAX_DIM * EKAN_NET_MAX_DIM];
    double v_skip[EKAN_NET_MAX_DIM * EKAN_NET_MAX_DIM];
} EkanNetLayer;

/* =========================================================================
 * EKAN Network
 * ========================================================================= */

typedef struct {
    EkanNetLayer layers[EKAN_NET_MAX_LAYERS];
    int topology[EKAN_NET_MAX_LAYERS + 1]; /* e.g., {12, 8, 4, 8, 12} */
    int num_layers;                         /* = len(topology) - 1 */
    int adam_step;                           /* global Adam step counter */
} EkanNetwork;

/* =========================================================================
 * Simple LCG PRNG for Glorot initialization (deterministic, no OS entropy)
 * ========================================================================= */

static inline double ekan_rand_normal(uint64_t *state) {
    /* Box-Muller from LCG */
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(*state >> 33) / (double)(1ULL << 31);
    if (u1 < 1e-10) u1 = 1e-10;
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(*state >> 33) / (double)(1ULL << 31);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* =========================================================================
 * Layer Init (Glorot/Xavier)
 * ========================================================================= */

static void ekan_layer_init(EkanNetLayer *layer, int in_dim, int out_dim,
                             uint64_t *rng_state) {
    memset(layer, 0, sizeof(EkanNetLayer));
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->gate_alpha = 0.0; /* σ(0) = 0.5: equal weight kan/skip */

    /* Glorot init for mixing weights */
    double limit = sqrt(6.0 / (in_dim + out_dim));
    for (int j = 0; j < out_dim; j++) {
        for (int i = 0; i < in_dim; i++) {
            layer->mixing[j][i] = ekan_rand_normal(rng_state) * limit;
        }
    }

    /* Glorot init for Fourier coefficients */
    double coeff_std = sqrt(2.0 / EKAN_NET_FOURIER_COEFFS);
    for (int j = 0; j < out_dim; j++) {
        for (int i = 0; i < in_dim; i++) {
            for (int c = 0; c < EKAN_NET_FOURIER_COEFFS; c++) {
                layer->fourier[j][i][c] = ekan_rand_normal(rng_state) * coeff_std;
            }
        }
    }

    /* Skip projection (only used when in_dim != out_dim) */
    if (in_dim != out_dim) {
        double skip_limit = sqrt(6.0 / (in_dim + out_dim));
        for (int k = 0; k < out_dim * in_dim; k++) {
            layer->skip_weights[k] = ekan_rand_normal(rng_state) * skip_limit;
        }
    }
}

/* =========================================================================
 * Layer Forward Pass
 * y_j = σ(α) × Σ_i[SiLU(W_ji·x_i) + Fourier(W_ji·x_i)]
 *      + (1-σ(α)) × skip(x)
 * ========================================================================= */

static void ekan_layer_forward(EkanNetLayer *layer,
                                const double *input, double *output) {
    int in_d = layer->in_dim;
    int out_d = layer->out_dim;

    /* Cache input for backward */
    for (int i = 0; i < in_d; i++)
        layer->last_input[i] = input[i];

    /* KAN output: Σ_i [SiLU(W_ji·x_i) + Fourier(W_ji·x_i)] */
    double kan_out[EKAN_NET_MAX_DIM];
    for (int j = 0; j < out_d; j++) {
        double sum = 0.0;
        for (int i = 0; i < in_d; i++) {
            double z = input[i] * layer->mixing[j][i]; /* entanglement */
            double act = ekan_silu(z) + ekan_fourier_eval(layer->fourier[j][i],
                                                           EKAN_NET_FOURIER_N, z);
            sum += act;
        }
        kan_out[j] = sum;
    }

    /* Residual gate: σ(α) */
    double gate = 1.0 / (1.0 + exp(-layer->gate_alpha));

    if (in_d == out_d) {
        /* Identity skip */
        for (int j = 0; j < out_d; j++)
            output[j] = gate * kan_out[j] + (1.0 - gate) * input[j];
    } else {
        /* Linear projection skip */
        for (int j = 0; j < out_d; j++) {
            double skip_j = 0.0;
            for (int i = 0; i < in_d; i++)
                skip_j += layer->skip_weights[j * in_d + i] * input[i];
            output[j] = gate * kan_out[j] + (1.0 - gate) * skip_j;
        }
    }
}

/* =========================================================================
 * Layer Backward Pass (Adam)
 *
 * Computes grad_input and updates all parameters using Adam optimizer.
 * ========================================================================= */

static void ekan_layer_backward_adam(EkanNetLayer *layer,
                                      const double *grad_output,
                                      double *grad_input,
                                      double lr, int step) {
    int in_d = layer->in_dim;
    int out_d = layer->out_dim;
    double beta1 = EKAN_ADAM_BETA1;
    double beta2 = EKAN_ADAM_BETA2;
    double eps = EKAN_ADAM_EPS;

    /* Bias correction */
    double bc1 = 1.0 - pow(beta1, step);
    double bc2 = 1.0 - pow(beta2, step);
    if (bc1 < 1e-10) bc1 = 1e-10;
    if (bc2 < 1e-10) bc2 = 1e-10;

    double gate = 1.0 / (1.0 + exp(-layer->gate_alpha));

    /* Recompute KAN outputs for gate gradient */
    double kan_out[EKAN_NET_MAX_DIM];
    for (int j = 0; j < out_d; j++) {
        double sum = 0.0;
        for (int i = 0; i < in_d; i++) {
            double z = layer->last_input[i] * layer->mixing[j][i];
            sum += ekan_silu(z) + ekan_fourier_eval(layer->fourier[j][i],
                                                     EKAN_NET_FOURIER_N, z);
        }
        kan_out[j] = sum;
    }

    /* Gate gradient */
    double grad_gate = 0.0;
    for (int j = 0; j < out_d; j++) {
        double skip_j;
        if (in_d == out_d) {
            skip_j = layer->last_input[j];
        } else {
            skip_j = 0.0;
            for (int i = 0; i < in_d; i++)
                skip_j += layer->skip_weights[j * in_d + i] * layer->last_input[i];
        }
        grad_gate += grad_output[j] * (kan_out[j] - skip_j) * gate * (1.0 - gate);
    }

    /* Scale grad for KAN path */
    double grad_kan[EKAN_NET_MAX_DIM];
    for (int j = 0; j < out_d; j++)
        grad_kan[j] = grad_output[j] * gate;

    /* Initialize grad_input */
    for (int i = 0; i < in_d; i++)
        grad_input[i] = 0.0;

    /* Per-edge gradients and updates */
    for (int j = 0; j < out_d; j++) {
        for (int i = 0; i < in_d; i++) {
            double z = layer->last_input[i] * layer->mixing[j][i];
            double dbase = ekan_silu_deriv(z);
            double dspline = ekan_fourier_deriv(layer->fourier[j][i],
                                                EKAN_NET_FOURIER_N, z);
            double dact = dbase + dspline;
            double grad_act = ekan_clip_grad(grad_kan[j] * dact, EKAN_MAX_GRAD_NORM);

            /* Mixing weight gradient */
            double g_mix = grad_act * layer->last_input[i];

            /* Adam update for mixing weight */
            layer->m_mixing[j][i] = beta1 * layer->m_mixing[j][i] + (1-beta1) * g_mix;
            layer->v_mixing[j][i] = beta2 * layer->v_mixing[j][i] + (1-beta2) * g_mix * g_mix;
            double m_hat = layer->m_mixing[j][i] / bc1;
            double v_hat = layer->v_mixing[j][i] / bc2;
            layer->mixing[j][i] -= lr * m_hat / (sqrt(v_hat) + eps);

            /* Accumulate grad_input */
            grad_input[i] += grad_act * layer->mixing[j][i];

            /* Skip path gradient to input */
            if (in_d == out_d) {
                if (i == j) grad_input[i] += grad_output[j] * (1.0 - gate);
            } else {
                grad_input[i] += grad_output[j] * (1.0 - gate) * layer->skip_weights[j * in_d + i];
            }

            /* Fourier coefficient gradients */
            double basis_vals[EKAN_NET_FOURIER_COEFFS];
            ekan_fourier_basis(EKAN_NET_FOURIER_N, z, basis_vals);
            for (int c = 0; c < EKAN_NET_FOURIER_COEFFS; c++) {
                double g_f = ekan_clip_grad(grad_kan[j] * basis_vals[c], EKAN_MAX_GRAD_NORM);

                /* L2 regularization on Fourier coeffs */
                g_f += EKAN_L2_LAMBDA * layer->fourier[j][i][c];

                /* Adam update */
                layer->m_fourier[j][i][c] = beta1 * layer->m_fourier[j][i][c] + (1-beta1) * g_f;
                layer->v_fourier[j][i][c] = beta2 * layer->v_fourier[j][i][c] + (1-beta2) * g_f * g_f;
                m_hat = layer->m_fourier[j][i][c] / bc1;
                v_hat = layer->v_fourier[j][i][c] / bc2;
                layer->fourier[j][i][c] -= lr * m_hat / (sqrt(v_hat) + eps);
            }
        }

        /* Skip weight updates (when dims differ) */
        if (in_d != out_d) {
            for (int i = 0; i < in_d; i++) {
                double g_s = grad_output[j] * (1.0 - gate) * layer->last_input[i];
                int idx = j * in_d + i;
                layer->m_skip[idx] = beta1 * layer->m_skip[idx] + (1-beta1) * g_s;
                layer->v_skip[idx] = beta2 * layer->v_skip[idx] + (1-beta2) * g_s * g_s;
                double m_hat2 = layer->m_skip[idx] / bc1;
                double v_hat2 = layer->v_skip[idx] / bc2;
                layer->skip_weights[idx] -= lr * m_hat2 / (sqrt(v_hat2) + eps);
            }
        }
    }

    /* Gate update (Adam) */
    grad_gate = ekan_clip_grad(grad_gate, EKAN_MAX_GRAD_NORM);
    layer->m_gate = beta1 * layer->m_gate + (1-beta1) * grad_gate;
    layer->v_gate = beta2 * layer->v_gate + (1-beta2) * grad_gate * grad_gate;
    {
        double m_hat = layer->m_gate / bc1;
        double v_hat = layer->v_gate / bc2;
        layer->gate_alpha -= lr * m_hat / (sqrt(v_hat) + eps);
    }
}

/* =========================================================================
 * Network API
 * ========================================================================= */

/* Initialize network with given topology, e.g. {12, 8, 4, 8, 12} */
static void ekan_net_init(EkanNetwork *net, const int *topology, int n_dims) {
    memset(net, 0, sizeof(EkanNetwork));
    net->num_layers = n_dims - 1;
    for (int i = 0; i <= net->num_layers; i++)
        net->topology[i] = topology[i];
    net->adam_step = 0;

    uint64_t rng = 42; /* deterministic seed */
    for (int l = 0; l < net->num_layers; l++) {
        ekan_layer_init(&net->layers[l], topology[l], topology[l+1], &rng);
    }
}

/* Forward prediction: chain forward() across all layers */
static void ekan_net_predict(const EkanNetwork *net,
                              const double *input, double *output) {
    double buf_a[EKAN_NET_MAX_DIM], buf_b[EKAN_NET_MAX_DIM];
    const double *cur_in = input;
    double *cur_out = buf_a;

    for (int l = 0; l < net->num_layers; l++) {
        /* Cast away const for layer forward (only writes last_input cache) */
        ekan_layer_forward((EkanNetLayer*)&net->layers[l], cur_in, cur_out);
        cur_in = cur_out;
        cur_out = (cur_out == buf_a) ? buf_b : buf_a;
    }

    /* Copy final output */
    int out_dim = net->topology[net->num_layers];
    for (int i = 0; i < out_dim; i++)
        output[i] = cur_in[i];
}

/* Online autoencoder update: forward → MSE grad → backward through all layers.
 * Returns MSE loss. */
static double ekan_net_update_online(EkanNetwork *net,
                                      const double *input,
                                      const double *target,
                                      double lr) {
    net->adam_step++;
    int out_dim = net->topology[net->num_layers];

    /* Forward pass (stores last_input in each layer) */
    double pred[EKAN_NET_MAX_DIM];
    ekan_net_predict(net, input, pred);

    /* MSE loss and output gradient */
    double loss = 0.0;
    double grad_out[EKAN_NET_MAX_DIM];
    for (int i = 0; i < out_dim; i++) {
        double err = pred[i] - target[i];
        loss += err * err;
        grad_out[i] = 2.0 * err / (double)out_dim;
    }
    loss /= (double)out_dim;

    /* Backward through all layers (reverse order) */
    double grad_a[EKAN_NET_MAX_DIM], grad_b[EKAN_NET_MAX_DIM];
    double *cur_grad_in = grad_out;
    double *cur_grad_out = grad_a;

    for (int l = net->num_layers - 1; l >= 0; l--) {
        ekan_layer_backward_adam(&net->layers[l], cur_grad_in, cur_grad_out,
                                 lr, net->adam_step);
        /* Swap buffers */
        double *tmp = cur_grad_in;
        cur_grad_in = cur_grad_out;
        cur_grad_out = (tmp == grad_a) ? grad_b :
                       (tmp == grad_b) ? grad_out : grad_a;
    }

    return loss;
}

/* Reconstruction error: ||input - reconstruct(input)||² / dims
 * Used for manifold anomaly detection. */
static double ekan_net_recon_error(EkanNetwork *net, const double *input) {
    int out_dim = net->topology[net->num_layers];
    double pred[EKAN_NET_MAX_DIM];
    ekan_net_predict(net, input, pred);

    double err = 0.0;
    for (int i = 0; i < out_dim; i++) {
        double d = input[i] - pred[i];
        err += d * d;
    }
    return err / (double)out_dim;
}

/* Sensitivity analysis: perturbation-based Jacobian column norms.
 * sens_out[d] = ||f(x + ε·eₐ) - f(x)|| for each input dimension d. */
static void ekan_net_sensitivity(EkanNetwork *net, const double *input,
                                  double *sens_out, double epsilon) {
    int in_dim = net->topology[0];
    int out_dim = net->topology[net->num_layers];

    double base_pred[EKAN_NET_MAX_DIM];
    ekan_net_predict(net, input, base_pred);

    double perturbed[EKAN_NET_MAX_DIM];
    double pert_pred[EKAN_NET_MAX_DIM];

    for (int d = 0; d < in_dim; d++) {
        for (int i = 0; i < in_dim; i++)
            perturbed[i] = input[i];
        perturbed[d] += epsilon;
        ekan_net_predict(net, perturbed, pert_pred);

        double delta = 0.0;
        for (int i = 0; i < out_dim; i++)
            delta += fabs(pert_pred[i] - base_pred[i]);
        sens_out[d] = delta;
    }
}

#endif /* MICROGPT_EKAN_NETWORK_H */
