import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Layer
from tensorflow.keras import Model, Input

def NNBlock_model(input_dims, NN_dims, activation="relu", name="NN"):
    x_inputs = Input(shape=(input_dims,), name="input_"+name)
    x = x_inputs
    for idx, layer_dim in enumerate(NN_dims):
        x = Dense(units=layer_dim, activation=activation,
                       name=name + "_dense_{}".format(idx))(x)

    return Model(inputs=x_inputs, outputs=x, name=name)


def NNBlockCond_model(input_dims, cond_dims, NN_dims, activation="relu", name="NN"):
    x_inputs = Input(shape=(input_dims,), name="input_" + name)
    cond_inputs = Input(shape=(cond_dims,), name="cond_input_" + name)

    x = concatenate([x_inputs, cond_inputs])

    for idx, layer_dim in enumerate(NN_dims):
        x = concatenate([Dense(units=layer_dim, activation=activation)(x), cond_inputs],
                             name=name + "_dense_cond_{}".format(idx))

    return Model(inputs=[x_inputs, cond_inputs], outputs=x, name=name)

def InceptionBlock_model(input_dims, NN_dims, activation="relu", name="NN"):
    x_inputs = Input(shape=(input_dims,), name="input_" + name)
    x = x_inputs
    for idx, layer_dim in enumerate(NN_dims):
        x = concatenate([Dense(units=layer_dim, activation=activation)(x), x],
                             name=name + "_dense_inception_{}".format(idx))

    return Model(inputs=x_inputs, outputs=x, name=name)

def EmbeddingBlock_model(input_dims, emb_dims, latent_dims, has_BN=False, activation="relu", name="NN"):
    embeddings = []
    cond_inputs= []

    for i, cond_d in enumerate(input_dims):
        c_inputs = Inputs(shape=(cond_d,), name = "input_cond_{}".format(i))
        cond_inputs.append(c_inputs)
        if emb_dims[i] == []:
            embeddings.append(c_inputs)
        else:
            first_emb = NNBlock_model(NN_dims=emb_dims[i], name=name + "cond_{}".format(i))
            embeddings.append(first_emb(c_inputs))

    concat_cond = concatenate(embeddings, name=name + "_emb_concat")

    last_emb = Dense(units=latent_dims, activation=activation, name=name + "_last_reduction")(
        concat_cond)

    if has_BN:
        last_emb = BatchNormalization()(last_emb)

    return Model(inputs=cond_inputs, outputs=last_emb, name=name)


@tf.function
def GaussianSampling(inputs):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the profil."""

    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon