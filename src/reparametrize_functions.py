import tensorflow as tf

@tf.function
def GaussianSampling(inputs):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the profil with a Gaussian distribution."""

    z_mean, z_log_var = inputs
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


