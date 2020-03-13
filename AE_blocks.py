import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Activation, average
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


def EmbeddingBlock_model(input_dims, emb_dims, has_BN=False, activation="relu", name="NN_emb"):
    embeddings = []
    not_to_emb =[]
    cond_inputs= []

    for i, cond_d in enumerate(input_dims):
        c_inputs = Input(shape=(cond_d,), name = "emb_input_cond_{}".format(i))
        cond_inputs.append(c_inputs)
        if len(emb_dims[i]) == 0:
            not_to_emb.append(c_inputs)
        else:
            first_emb = AE_blocks(input_dims=cond_d, NN_dims=emb_dims[i], type="NNBlock_model",
                                  name=name + "cond_{}".format(i))
            emb_cond = first_emb(c_inputs)
            if has_BN:
                emb_cond = BatchNormalization()(emb_cond)
            embeddings.append(emb_cond)

    all_embs = concatenate(embeddings, name=name + "_emb_concat")

    last_emb = Dense(units=emb_dims[-1], activation=None, name=name + "_last_reduction")(all_embs)

    if has_BN:
        last_emb = BatchNormalization()(last_emb)

    emb_outputs = Activation(activation)(last_emb)

    if len(not_to_emb) != 0:
        emb_outputs = concatenate([emb_outputs] + not_to_emb)

    return Model(inputs=cond_inputs, outputs=emb_outputs, name=name)

def build_encoder_model(self, model_params):

    x_inputs = Input(shape=(model_params.input_dims,), name="enc_inputs")
    c_inputs = []
    ensemble=[[] for i in range(model_params.nb_latent_components)]

    cond_enc_inputs_dims=[]

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="enc_cond_inputs_{}".format(i)))

    inputs = [x_inputs] + c_inputs

    # Creation of the encoder block
    if len(c_inputs)>=1:
        if model_params.with_embedding:
            self.to_embedding = EmbeddingBlock_model(input_dims=model_params.cond_dims, emb_dims=model_params.emb_dims,
                                               activation="relu",name="emb", has_BN=False)
            cond_enc_inputs = self.to_embedding(c_inputs)
        else:
            if len(c_inputs) >=2:
                cond_enc_inputs = concatenate(c_inputs, name="concat_cond")
            else:
                cond_enc_inputs = c_inputs
        cond_enc_inputs_dims.append(K.int_shape(cond_enc_inputs[0])[-1])
        enc_inputs = [x_inputs, cond_enc_inputs]
    else:
        enc_inputs = [x_inputs]

    for idx in tf.range(0, model_params.nb_encoder_ensemble, 1):
        encoder_block = AE_blocks(input_dims=model_params.input_dims, cond_dims=cond_enc_inputs_dims,
                                  type=model_params.encoder_type, NN_dims=model_params.encoder_dims,
                                  name="encoder_block_{}".format(idx), activation="relu")
        enc_x = encoder_block(enc_inputs)

        for i in tf.range(0, model_params.nb_latent_components, 1):
            ensemble[i].append(Dense(units=model_params.latent_dims, activation='linear',
                                     name="latent_dense_{}_{}".format(idx,i+1))(enc_x))

    if model_params.nb_encoder_ensemble ==1:
        enc_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        enc_outputs = [average(ens_list, name="enc_averaging") for ens_list in ensemble]

    return Model(inputs=inputs, outputs=enc_outputs, name="encoder")

def build_decoder_model(self, model_params):

    encoded_inputs = Input(shape=(model_params.latent_dims,), name="encoded_inputs")
    c_inputs = []
    ensemble = [[] for i in range(model_params.nb_decoder_outputs)]
    cond_dec_inputs_dims = []

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="dec_cond_inputs_{}".format(i)))

    inputs = [encoded_inputs] + c_inputs

    if len(c_inputs)>=1:
        if model_params.with_embedding:
            cond_dec_inputs = self.to_embedding(c_inputs)
        else:
            if len(c_inputs) >=2:
                cond_dec_inputs = concatenate(c_inputs, name="concat_cond")
            else:
                cond_dec_inputs = c_inputs
        cond_dec_inputs_dims.append(K.int_shape(cond_dec_inputs[0])[-1])
        dec_inputs = [encoded_inputs, cond_dec_inputs]
    else:
        dec_inputs = [encoded_inputs]

    for idx in tf.range(0, model_params.nb_decoder_ensemble, 1):
        decoder_block = AE_blocks(input_dims=model_params.latent_dims, cond_dims=cond_dec_inputs_dims,
                                  type=model_params.decoder_type, NN_dims=model_params.decoder_dims,
                                  name="decoder_block_{}".format(idx), activation="relu")

        dec_x = decoder_block(dec_inputs)

        for i in tf.range(0, model_params.nb_decoder_outputs, 1):
            ensemble[i].append(Dense(units=model_params.input_dims, activation='linear',
                                     name="dec_output_{}_{}".format(idx,i+1))(dec_x))

    if model_params.nb_decoder_ensemble == 1:
        dec_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        dec_outputs = [average(ens_list, name="dec_averaging") for ens_list in ensemble]

    return Model(inputs=inputs, outputs=dec_outputs, name="decoder")

