# A file where to define losses
import tensorflow as tf, numpy as np
from tensorflow.keras import backend as K


@tf.function
def build_kl_loss(y_true, y_pred, latent_components, prior_mu=K.variable(0.), log_prior_sigma=K.variable(0.),
                  annealing_value = K.variable(0.)):
    latent_mu, latent_log_sigma = latent_components
    kl = K.mean(0.5 * K.sum((K.exp(latent_log_sigma) + K.square(latent_mu - prior_mu)) / K.exp(log_prior_sigma)
                       - 1. - latent_log_sigma + log_prior_sigma, axis=-1))
    return K.abs(kl - annealing_value)

@tf.function
def build_gaussian_loglikelihood(y_true, y_pred, log_sigma=K.variable(0.)):
    return K.mean(0.5 * (log_sigma + K.square(y_true-y_pred) / K.exp(log_sigma)))

@tf.function
def kde(s1, s2, h=None):
    dim = K.shape(s1)[1]
    s1_size = K.shape(s1)[0]
    s2_size = K.shape(s2)[0]
    if h is None:
        h = K.cast(dim, dtype='float32') / 2
    tiled_s1 = K.tile(K.reshape(s1, K.stack([s1_size, 1, dim])), K.stack([1, s2_size, 1]))
    tiled_s2 = K.tile(K.reshape(s2, K.stack([1, s2_size, dim])), K.stack([s1_size, 1, 1]))
    return K.exp(-0.5 * K.sum(K.square(tiled_s1 - tiled_s2), axis=-1) / h)


@tf.function
def build_mmd_loss(y_true, y_pred, latent_mu, latent_sampling):
    q_kernel = kde(latent_mu, latent_mu)
    p_kernel = kde(latent_sampling, latent_sampling)
    pq_kernel = kde(latent_mu, latent_sampling)
    return K.mean(q_kernel) + K.mean(p_kernel) - 2 * K.mean(pq_kernel)


def build_gram_matrix(s1, h):
    s1 = K.cast(s1, dtype='float64')
    h = K.cast(h, dtype='float64')

    dim = K.shape(s1)[1]
    s1_size = K.shape(s1)[0]

    tiled_s1 = K.tile(K.reshape(s1, K.stack([s1_size, 1, dim])), K.stack([1, s1_size, 1]))
    tiled_s2 = K.tile(K.reshape(s1, K.stack([1, s1_size, dim])), K.stack([s1_size, 1, 1]))
    gram_matrix = K.exp(-0.5 * K.sum(K.square(K.cast((tiled_s1 - tiled_s2), dtype='float64')), axis=-1) / h)

    diag = tf.linalg.diag_part(gram_matrix)
    diag_1 = K.tile(K.reshape(diag, K.stack([s1_size, 1])), K.stack([1, s1_size]))
    diag_2 = K.tile(K.reshape(diag, K.stack([1, s1_size])), K.stack([s1_size, 1]))
    normed_gram_matrix = gram_matrix / K.sqrt(diag_1 * diag_2) / (K.cast(s1_size, dtype='float64'))

    return normed_gram_matrix
    #TODO: use tf.norm ou tfp.kernel pour calculer la matrice

def trace_normalize(A):
    trace_A = tf.linalg.trace(A)
    return A / trace_A


def Renyi_entropy(A, alpha):
    A_eigval,v = tf.linalg.eigh(A)
    Rent = K.log(K.sum(K.pow(A_eigval+1e-8, alpha))) / K.cast((1. - alpha) * K.log(2.), dtype='float64')
    return K.cast(Rent, dtype='float32')


def build_mutualinfo_loss(y_true, y_pred, cond_true, latent_mu, sigma=3, alpha=1.01, kappa=10.):

    if len(cond_true)>=2:
        cond_in = K.concatenate(cond_true, axis=-1)
    else:
        cond_in = cond_true[0]

    gram_x = build_gram_matrix(y_true, sigma)
    gram_c = build_gram_matrix(cond_in, sigma)
    gram_z = build_gram_matrix(latent_mu, sigma)

    cond_MI = Renyi_entropy(trace_normalize(gram_z), alpha) + \
              Renyi_entropy(trace_normalize(gram_c), alpha) -\
              Renyi_entropy(trace_normalize(gram_c * gram_z), alpha)

    input_MI = Renyi_entropy(trace_normalize(gram_z), alpha) + \
               Renyi_entropy(trace_normalize(gram_x), alpha) -\
               Renyi_entropy(trace_normalize(gram_x * gram_z), alpha)

    return K.cast(1. / kappa / input_MI + kappa * cond_MI, dtype='float32')