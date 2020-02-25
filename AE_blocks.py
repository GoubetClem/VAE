import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Layer
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K

class AE_blocks():

    def __init__(self, input_dims, NN_dims, type="NNBlock_model", activation="relu", name="NN", **kwargs):
        self.input_dims = input_dims
        self.NN_dims = NN_dims
        self.activation = activation
        self.name= name
        self.block = getattr(self, type)(**kwargs)

    def __call__(self, inputs):
        return self.block(inputs)


    def NNBlock_model(self, **kwargs):
        x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
        x = x_inputs
        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation,
                           name=self.name + "_dense_{}".format(idx))(x)

        return Model(inputs=x_inputs, outputs=x, name=self.name)


    def NNBlockCond_model(self, cond_dims, **kwargs):
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)

        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx))(x)
            for c_input in cond_inputs:
                x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)
    # TODO TargetNNBlockCond et Conv1D starting block

    def InceptionBlock_model(self, cond_dims, **kwargs):
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)
        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = concatenate([Dense(units=layer_dim, activation=self.activation)(x), x],
                                 name=self.name + "_dense_inception_{}".format(idx))

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

def EmbeddingBlock_model(self, input_dims, emb_dims, latent_dims, has_BN=False, activation="relu", name="NN"):
    embeddings = []
    cond_inputs= []

    for i, cond_d in enumerate(input_dims):
        c_inputs = Input(shape=(cond_d,), name = "input_cond_{}".format(i))
        cond_inputs.append(c_inputs)
        if emb_dims[i] == []:
            embeddings.append(c_inputs)
        else:
            first_emb = AE_blocks(NN_dims=emb_dims[i], type="NNBlock_model", name=name + "cond_{}".format(i))
            embeddings.append(first_emb(c_inputs))

    concat_cond = concatenate(embeddings, name=name + "_emb_concat")

    last_emb = Dense(units=latent_dims, activation=activation, name=name + "_last_reduction")(
        concat_cond)

    if has_BN:
        last_emb = BatchNormalization()(last_emb)

    return Model(inputs=cond_inputs, outputs=last_emb, name=name)

def encoder_model(self, type, input_dims, latent_dims, encoder_dims=[24], number_outputs=1, **kwargs):

    x_inputs = Input(shape=(input_dims,), name="x_inputs")
    c_inputs = []
    enc_outputs=[]
    cond_enc_inputs_dims=[]

    for i, c_dims in enumerate(self.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

    inputs = [x_inputs] + c_inputs

    # Creation of the encoder block
    if self.with_embedding:
        self.to_embedding = EmbeddingBlock_model(input_dims=self.cond_dims, emb_dims=self.emb_dims,
                                                 latent_dims=latent_dims, activation="relu",
                                                 name="emb", has_BN=True)
        cond_enc_inputs = self.to_embedding(c_inputs)
        cond_enc_inputs_dims.append(latent_dims)
        enc_inputs = [x_inputs, cond_enc_inputs]
    elif len(self.cond_dims) >= 2:
        cond_enc_inputs = concatenate(c_inputs, name="concat_cond")
        cond_enc_inputs_dims.append(K.int_shape(cond_enc_inputs)[-1])
        enc_inputs = [x_inputs, cond_enc_inputs]
    elif len(self.cond_dims) == 1:
        cond_enc_inputs_dims.append(K.int_shape(c_inputs[0])[-1])
        enc_inputs = [x_inputs, c_inputs]
    else:
        enc_inputs = [x_inputs]

    encoder_block = AE_blocks(input_dims=input_dims, cond_dims=cond_enc_inputs_dims, type=type, NN_dims=encoder_dims,
                              name="encoder_block", activation="relu")
    enc_x = encoder_block(enc_inputs)

    for i in tf.range(0, number_outputs, 1):
        enc_outputs.append(Dense(units=latent_dims, activation='linear', name="latent_dense_{}".format(i+1))(enc_x))

    return Model(inputs=inputs, outputs=enc_outputs, name="encoder")

def decoder_model(self, type, input_dims, latent_dims, decoder_dims=[24], **kwargs):

    encoded_inputs = Input(shape=(latent_dims,), name="encoded_inputs")
    c_inputs = []
    cond_dec_inputs_dims = []

    for i, c_dims in enumerate(self.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

    inputs = [encoded_inputs] + c_inputs

    if self.with_embedding:
        cond_dec_inputs = self.to_embedding(c_inputs)
        cond_dec_inputs_dims.append(latent_dims)
        dec_inputs = [encoded_inputs, cond_dec_inputs]
    elif len(self.cond_dims) >= 2:
        cond_dec_inputs = concatenate(c_inputs, name="concat_cond")
        cond_dec_inputs_dims.append(K.int_shape(cond_dec_inputs)[-1])
        dec_inputs = [encoded_inputs, cond_dec_inputs]
    elif len(self.cond_dims) == 1:
        cond_dec_inputs_dims.append(K.int_shape(c_inputs[0])[-1])
        dec_inputs = [encoded_inputs, c_inputs]
    else:
        dec_inputs = [encoded_inputs]

    decoder_block = AE_blocks(input_dims=latent_dims, cond_dims=cond_dec_inputs_dims, type=type, NN_dims=decoder_dims,
                              name="decoder_block", activation="relu")
    dec_x = decoder_block(dec_inputs)

    dec_outputs = Dense(input_dims, activation='linear', name='dec_output')(dec_x)

    return Model(inputs=inputs, outputs=dec_outputs, name="decoder")


@tf.function
def GaussianSampling(inputs):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding the profil."""

    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon