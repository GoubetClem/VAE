import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Layer

class NNBlock(tf.keras.layers.Layer):

    def __init__(self, NN_dims, name_block="NN", activation="relu"):

        super(NNBlock, self).__init__()
        self.NN_dims = NN_dims
        self.name_block = name_block
        self.activation = activation

    def call(self, inputs):
        for  idx, layer_dim in enumerate(self.NN_dims):
            inputs = Dense(units=layer_dim, activation=self.activation,
                           name=self.name_block+"_dense_{}".format(idx))(inputs)
            return inputs

class NNBlockCond(Layer):

    def __init__(self, NN_dims, name_block="NN", activation="relu"):
        super(NNBlockCond, self).__init__()
        self.NN_dims = NN_dims
        self.name_block = name_block
        self.activation = activation

    def call(self, inputs, cond_inputs):
        for idx, layer_dim in enumerate(self.NN_dims):
            inputs = concatenate([Dense(units=layer_dim, activation=self.activation)(inputs), cond_inputs],
                                name=self.name_block+"_dense_cond_{}".format(idx))
            return inputs

class ResnetBlock(Layer):

    def __init__(self, NN_dims, name_block="NN", activation="relu"):
        super(ResnetBlock, self).__init__()
        self.NN_dims = NN_dims
        self.name_block = name_block
        self.activation = activation

    def call(self, inputs):
        for  idx, layer_dim in enumerate(self.NN_dims):
            inputs = concatenate([Dense(units=layer_dim, activation=self.activation)(inputs), inputs],
                                 name=self.name_block + "_dense_resnet_{}".format(idx))
            return inputs

class EmbeddingBlock(Layer):

    def __init__(self, emb_dims, latent_dims, activation="relu", name_block="NN", has_BN=True):
        super(EmbeddingBlock, self).__init__()
        self.emb_dims = emb_dims
        self.latent_dims = latent_dims
        self.activation = activation
        self.name_block = name_block
        self.has_BN = has_BN

    def call(self, cond_inputs):
        embeddings = []

        for i, cond in enumerate(cond_inputs):
            if self.emb_dims[i] == 0:
                embeddings.append(cond)
            else:
                first_emb = NNBlock(NN_dims=self.emb_dims[i], name=self.name_block+"cond_{}".format(i))
                embeddings.append(first_emb(cond))

        concat_cond = concatenate(embeddings, name=self.name_block+"_emb_concat")

        last_emb = Dense(units=self.latent_dims, activation = self.activation, name=self.name_block+"_last_reduction")(concat_cond)

        if self.has_BN:
            last_emb = BatchNormalization()(last_emb)

        return last_emb

class GaussianSampling(Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding the profil."""

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon