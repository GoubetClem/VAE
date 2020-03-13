# A file where to define losses
import tensorflow as tf
from tensorflow.keras import backend as K

@tf.function
def build_kl_loss(y_true, y_pred, latent_components, prior_mu=K.variable(0.), log_prior_sigma=K.variable(0.)):
    latent_mu, latent_log_sigma = latent_components
    return K.mean(0.5 * K.sum((K.exp(latent_log_sigma) + K.square(latent_mu - prior_mu)) / K.exp(log_prior_sigma)
                       - 1. - latent_log_sigma + log_prior_sigma, axis=-1))

@tf.function
def build_gaussian_loglikelihood(y_true, y_pred, log_sigma=K.variable(0.)):
    return K.mean(K.sum(-0.5 * (log_sigma + K.square(y_true-y_pred) / K.exp(log_sigma)), axis=-1))

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
    K.cast(s1, dtype='float64')
    K.cast(h, dtype='float64')

    dim = K.shape(s1)[1]
    s1_size = K.shape(s1)[0]

    tiled_s1 = K.tile(K.reshape(s1, K.stack([s1_size, 1, dim])), K.stack([1, s1_size, 1]))
    tiled_s2 = K.tile(K.reshape(s1, K.stack([1, s1_size, dim])), K.stack([s1_size, 1, 1]))
    Dist_M = K.exp(-0.5 * K.square(K.cast((tiled_s1 - tiled_s2), dtype='float64') / h))

    list_dist_M = [K.squeeze(K.slice(Dist_M, [0, 0, i], K.stack([s1_size, s1_size, 1])), axis=-1) for i in
                   range(K.int_shape(s1)[1])]

    gram_list = []
    for dist_M in list_dist_M:
        diag = tf.diag_part(dist_M)
        diag_1 = K.tile(K.reshape(diag, K.stack([s1_size, 1])), K.stack([1, s1_size]))
        diag_2 = K.tile(K.reshape(diag, K.stack([1, s1_size])), K.stack([s1_size, 1]))
        gram_list.append(dist_M / K.sqrt(diag_1 * diag_2) / K.cast(s1_size, dtype='float64'))

    return gram_list


def build_joint_entropy(list_M):
    AB = K.cast(K.ones_like(list_M[0]), dtype='float64')
    for M in list_M:
        AB = AB * M

    res = AB / K.cast(K.reshape(tf.trace(AB), K.stack([1])), dtype='float64')
    return K.reshape(res, K.shape(list_M[0]))


def build_Renyi_entropy(A, alpha):
    A = K.cast(A, dtype='float64')
    # A = K.switch(K.sum(K.cast(tf.is_nan(A),dtype='float64')) >0, K.ones(1, dtype='float64'), A)
    input_A = K.cast(0.5 * (A + K.transpose(A)), dtype='float64')
    eig_values, _ = tf.linalg.eigh(input_A)
    eig_values = eig_values + 1.e-6
    return K.cast(K.log(K.sum(K.pow(eig_values, alpha))), dtype='float64') / K.cast((1. - alpha) * K.log(2.),
                                                                                    dtype='float64')


def build_entropy_loss(y_true, y_pred, cond_true, latent_mu, h=5, alpha=1.01, kappa=K.variable(1.)):
    #z = K.switch(K.greater(recon_loss(y_true, y_pred), 100.), K.random_normal(K.shape(z_mu), seed=42), z_mu)
    sigma = h * K.pow(K.cast(K.shape(y_true)[0], dtype='float64'), -1. / (4. + 1.))
    gram_x = build_gram_matrix(y_true, sigma)
    gram_c = build_gram_matrix(cond_true, sigma)
    gram_z = build_gram_matrix(latent_mu, sigma)  # K.ones_like(z_mu, dtype='float32')
    # gram_z = [K.eye(self.batch_size, dtype='float64')*K.cast(self.buffer < 10000, dtype='float64')]*4 #+ K.cast(self.buffer > 10000, dtype='float64')*g_z for g_z in gram_z] #]

    joint_entropy_x = build_joint_entropy(gram_x)
    joint_entropy_z = build_joint_entropy(gram_z)
    joint_entropy_cond = build_joint_entropy(gram_c)
    joint_entropy_xz = build_joint_entropy(gram_z + gram_x)
    joint_entropy_zcond = build_joint_entropy(gram_z + gram_c)

    cond_MI = build_Renyi_entropy(joint_entropy_z, alpha) + \
              build_Renyi_entropy(joint_entropy_cond, alpha) -\
              build_Renyi_entropy(joint_entropy_zcond, alpha)

    input_MI = build_Renyi_entropy(joint_entropy_z, alpha) + \
               build_Renyi_entropy(joint_entropy_x, alpha) -\
               build_Renyi_entropy(joint_entropy_xz, alpha)

    return 1 / input_MI + kappa  * cond_MI