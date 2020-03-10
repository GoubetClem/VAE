import tensorflow as tf

@tf.function
def GaussianSampling(inputs):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the profil."""

    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def GaussianMixtureSampling(inputs):

    mu_tensor, log_sigma_tensor, mixture_vec = inputs
    batch, dim, n_mixture = tf.shape(mu_tensor)

    epsilon = tf.keras.backend.random_normal(shape=(batch, dim, n_mixture))
