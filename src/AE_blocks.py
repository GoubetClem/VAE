from matplotlib import units
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Activation, average,\
    Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Add, Multiply, Lambda, LSTM
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from src.reparametrize_functions import *


class AE_blocks():
    """
    Class that will initiate and save a block of tf.keras layers. Different architectures off the shelves.
    """

    def __init__(self, input_dims, NN_dims, type="NNBlock_model", activation="relu", name="NN", 
                 kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', conditions_layer=[], **kwargs):
        self.input_dims = input_dims #input dims of the block
        self.NN_dims = NN_dims #list of hidden layers dims
        self.activation = activation #layer activation to consider for each layer of the block
        self.name= name #name for the block
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.conditions_layer = conditions_layer
        self.block = getattr(self, type)(**kwargs)  
        

    def __call__(self, inputs):
        return self.block(inputs)


    def NNBlock_model(self, **kwargs):
        """

        :param kwargs: Necessary arguments
        :return: A sequential feedforward block of Dense layers
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
        x = x_inputs
        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation,
                           name=self.name + "_dense_{}".format(idx),
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer = self.bias_initializer)(x)

        return Model(inputs=x_inputs, outputs=x, name=self.name)

    def NNBlock_binary_model(self, **kwargs):
        """

        :param kwargs: Necessary arguments
        :return: A sequential feedforward block of Dense layers
        """

        def softmax(x):
            return tf.math.exp(x)/tf.reshape(K.sum(tf.exp(x), axis=1),(-1,1))

        x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
        x = x_inputs
        for idx, layer_dim in enumerate(self.NN_dims):
            if (idx<len(self.NN_dims)-1):
                x = Dense(units=layer_dim, activation=self.activation,
                           name=self.name + "_dense_{}".format(idx),
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer = self.bias_initializer)(x)
            else: 
                x = Dense(units=layer_dim, activation='tanh',
                           name=self.name + "_dense_{}".format(idx),
                           kernel_initializer=self.kernel_initializer,
                           bias_initializer = self.bias_initializer)(x)
                #x = Lambda(softmax)(x)

        return Model(inputs=x_inputs, outputs=x, name=self.name)

    def NNBlock_Unbiaised_binary_model(self, **kwargs):
        """

        :param kwargs: Necessary arguments
        :return: A sequential feedforward block of Dense layers
        """

        def softmax(x):
            return tf.math.exp(x)/tf.reshape(K.sum(tf.exp(x), axis=1),(-1,1))

        x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
        x = x_inputs
        for idx, layer_dim in enumerate(self.NN_dims):
            if (idx<len(self.NN_dims)-1):
                x = Dense(units=layer_dim, activation=self.activation,
                           name=self.name + "_dense_{}".format(idx),
                           kernel_initializer=self.kernel_initializer,
                           use_bias = False)(x)
            else: 
                x = Dense(units=layer_dim, activation='tanh',
                           name=self.name + "_dense_{}".format(idx),
                           kernel_initializer=self.kernel_initializer,
                           use_bias = False)(x)
                #x = Lambda(softmax)(x)

        return Model(inputs=x_inputs, outputs=x, name=self.name)

    def NNBlock_Unbiaised_model(self, **kwargs):
            """

            :param kwargs: Necessary arguments
            :return: A sequential feedforward block of Dense layers
            """
            x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
            x = x_inputs
            for idx, layer_dim in enumerate(self.NN_dims):
                if (idx<len(self.NN_dims)-1):
                    x = Dense(units=layer_dim, activation=self.activation,
                                name=self.name + "_dense_{}".format(idx),
                                kernel_initializer=self.kernel_initializer,
                                use_bias=False)(x)
                else:
                    x = Dense(units=layer_dim, activation='linear', name=self.name + "_dense_cond_{}".format(idx), 
                                                kernel_initializer=self.kernel_initializer, 
                                                use_bias = False)(x)

            return Model(inputs=x_inputs, outputs=x, name=self.name)

    def NNBlock_perturbation_model(self, **kwargs):
            """

            :param kwargs: Necessary arguments
            :return: A sequential feedforward block of Dense layers
            """
            x_inputs = Input(shape=(self.input_dims,), name="input_"+self.name)
            x = x_inputs
            for idx, layer_dim in enumerate(self.NN_dims):
                if (idx<len(self.NN_dims)-1):
                    x = Dense(units=layer_dim, activation=self.activation,
                                name=self.name + "_dense_{}".format(idx),
                                kernel_initializer=self.kernel_initializer,
                                bias_initializer = self.bias_initializer)(x)
                else:
                    x = Dense(units=layer_dim, activation='linear', name=self.name + "_dense_cond_{}".format(idx), 
                                                kernel_initializer=self.kernel_initializer, 
                                                bias_initializer = self.bias_initializer)(x)

            return Model(inputs=x_inputs, outputs=x, name=self.name)
            

    def NNBlockCond_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A sequential feedforward Dense block with recall of the cond inputs at each layer
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)
 
        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx), kernel_initializer=self.kernel_initializer, 
                                                                                                                  bias_initializer = self.bias_initializer)(x)
            for c_input in cond_inputs:
                x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

    
    def NNBlockControlCond_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A sequential feedforward Dense block with recall of the cond inputs at each layer
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)

        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx), kernel_initializer=self.kernel_initializer, 
                                                                                                                  bias_initializer = self.bias_initializer)(x)
            if self.conditions_layer[idx]:
                for c_input in cond_inputs:
                    x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

    def NNBlockCond_Unbiaised_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A sequential feedforward Dense block with recall of the cond inputs at each layer
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)

        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx), 
                      kernel_initializer=self.kernel_initializer, use_bias=False)(x)
            for c_input in cond_inputs:
                x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

    def NNBlockConv1D_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A sequential feedforward block, first applying a Conv1D layer on inputs, then a Dense block with recall of conds at each layers
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)

        x = Conv1D(filters=self.input_dims, kernel_size=4, strides=1, padding="causal",
                   name="conv1D_layer")(tf.expand_dims(x_inputs, axis=-1))

        x = Conv1D(filters=self.input_dims, kernel_size=10, strides=1,
                   name="conv1D_layer_2")(x)

        x = MaxPooling1D(pool_size= 3, name="pooling_layer")(x)
        x = Flatten(name = "global_ravel")(x)

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx))(x)
            for c_input in cond_inputs:
                x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)


    def NNBlockLSTM_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A sequential feedforward block, first applying a Conv1D layer on inputs, then a Dense block with recall of conds at each layers
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)

        x = LSTM(units=self.input_dims, activation=self.activation, recurrent_activation="tanh",
                 dropout=0.1, recurrent_dropout=0.05, return_sequences=False,
                 name="LSTM_layer")(tf.expand_dims(x_inputs, axis=-1))

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = Dense(units=layer_dim, activation=self.activation, name=self.name + "_dense_cond_{}".format(idx))(x)
            for c_input in cond_inputs:
                x = concatenate([x, c_input])

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)


    def InceptionBlock_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A feedforward block, with recall of each previous layers outputs
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)
        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = concatenate([Dense(units=layer_dim, activation=self.activation, 
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer = self.bias_initializer)(x), x],
                                 name=self.name + "_dense_inception_{}".format(idx))

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

    def InceptionBlock_oldmodel(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A feedforward block, with recall of each previous layers outputs
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)
        x = x_inputs

        cond_inputs = []
        cond_inputs.append(Input(shape=(cond_dims[0],), name="cond_input_" + self.name))
        x = concatenate([x_inputs, cond_inputs[-1]])

    

        for idx, layer_dim in enumerate(self.NN_dims):
            x = concatenate([Dense(units=layer_dim, activation=self.activation, 
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer = self.bias_initializer)(x), x],
                                 name=self.name + "_dense_inception_{}".format(idx))

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)

    def InceptionBlock_Unbiaised_model(self, cond_dims, **kwargs):
        """

        :param cond_dims: list of each condition inputs dimensions
        :param kwargs: Necessary arguments
        :return: A feedforward block, with recall of each previous layers outputs
        """
        x_inputs = Input(shape=(self.input_dims,), name="input_" + self.name)
        x = x_inputs

        cond_inputs = []

        for c_dims in cond_dims:
            cond_inputs.append(Input(shape=(c_dims,), name="cond_input_" + self.name))
            x = concatenate([x, cond_inputs[-1]])

        for idx, layer_dim in enumerate(self.NN_dims):
            x = concatenate([Dense(units=layer_dim, activation=self.activation, 
                                   kernel_initializer=self.kernel_initializer,
                                   use_bias = False)(x), x],
                                 name=self.name + "_dense_inception_{}".format(idx))

        return Model(inputs=[x_inputs] + cond_inputs, outputs=x, name=self.name)


def EmbeddingBlock_model(input_dims, emb_dims, kernel_init, bias_init, has_BN=False, activation="relu", name="NN_emb"):
    """

    :param input_dims: list, list of dimensions of the tensors ton consider
    :param emb_dims: list, list of lists to specify the NN block of each input. Empty if not to be embedded.
    :param has_BN: Boolean, to add batch normalization
    :param activation: str, layer activation parameter. Default = "relu"
    :param name: "name of the block"
    :return: TF Model, with embedded inputs concatenated with not changed inputs
    """
    embeddings = []
    not_to_emb = []
    cond_inputs = []

    for i, cond_d in enumerate(input_dims):
        c_inputs = Input(shape=(cond_d,), name="emb_input_cond_{}".format(i))
        cond_inputs.append(c_inputs)
        if len(emb_dims[i]) == 0:
            not_to_emb.append(c_inputs)
        else:
            first_emb = AE_blocks(input_dims=cond_d, NN_dims=emb_dims[i], type="NNBlock_model",
                                  kernel_initializer=kernel_init,
                                  bias_initializer=bias_init,
                                  name=name + "cond_{}".format(i))
            emb_cond = first_emb(c_inputs)
            if has_BN:
                emb_cond = BatchNormalization()(emb_cond)
            embeddings.append(emb_cond)

    all_embs = concatenate(embeddings, name=name + "_emb_concat")

    last_emb = Dense(units=emb_dims[-1], activation=None, name=name + "_last_reduction", kernel_initializer=kernel_init, 
                                                                                         bias_initializer = bias_init)(all_embs)

    if has_BN:
        last_emb = BatchNormalization()(last_emb)

    emb_outputs = Activation(activation)(last_emb)

    if len(not_to_emb) != 0:
        emb_outputs = concatenate([emb_outputs] + not_to_emb)

    return Model(inputs=cond_inputs, outputs=emb_outputs, name=name)

def EmbeddingBlock_Unbiaised_model(input_dims, emb_dims,kernel_init, has_BN=False, activation="relu", name="NN_emb"):
    """

    :param input_dims: list, list of dimensions of the tensors ton consider
    :param emb_dims: list, list of lists to specify the NN block of each input. Empty if not to be embedded.
    :param has_BN: Boolean, to add batch normalization
    :param activation: str, layer activation parameter. Default = "relu"
    :param name: "name of the block"
    :return: TF Model, with embedded inputs concatenated with not changed inputs
    """
    embeddings = []
    not_to_emb = []
    cond_inputs = []

    for i, cond_d in enumerate(input_dims):
        c_inputs = Input(shape=(cond_d,), name="emb_input_cond_{}".format(i))
        cond_inputs.append(c_inputs)
        if len(emb_dims[i]) == 0:
            not_to_emb.append(c_inputs)
        else:
            first_emb = AE_blocks(input_dims=cond_d, NN_dims=emb_dims[i], type="NNBlock_Unbiaised_model",kernel_initializer=kernel_init,
                                  name=name + "cond_{}".format(i))
            emb_cond = first_emb(c_inputs)
            if has_BN:
                emb_cond = BatchNormalization()(emb_cond)
            embeddings.append(emb_cond)

    all_embs = concatenate(embeddings, name=name + "_emb_concat")

    last_emb = Dense(units=emb_dims[-1], activation=None, name=name + "_last_reduction",kernel_initializer=kernel_init, use_bias=False)(all_embs)

    if has_BN:
        last_emb = BatchNormalization()(last_emb)

    emb_outputs = Activation(activation)(last_emb)

    if len(not_to_emb) != 0:
        emb_outputs = concatenate([emb_outputs] + not_to_emb)

    return Model(inputs=cond_inputs, outputs=emb_outputs, name=name)

def build_embedding_model(self, model_params, name=""):
    input_dims=model_params.cond_dims
    emb_dims=model_params.emb_dims
    if model_params.embedding_type == 'EmbeddingBlock_Unbiaised_model':
        return EmbeddingBlock_Unbiaised_model(input_dims=input_dims, emb_dims=emb_dims, kernel_init=model_params.kernel_initializer, 
                                                                                        activation='relu', name='cond_emb', has_BN=False)

    return EmbeddingBlock_model(input_dims=input_dims, emb_dims=emb_dims, kernel_init=model_params.kernel_initializer, 
                                bias_init=model_params.bias_initializer ,activation='relu', name='cond_emb', has_BN=False)

def build_leap_block(latent_dims, context_dims, block_dims, name="leap_block"):
    latent_inputs = Input(shape=(latent_dims,), name="latentcoord_inputs")
    context_inputs = Input(shape=(context_dims,), name="context_inputs")
    block_inputs = [latent_inputs, context_inputs]
    latent_coord = latent_inputs

    for i, tau_dims in enumerate(block_dims):
        latent_coord = Dense(units=tau_dims, activation="relu", name="enc_context_{}".format(i))(latent_coord)
    leap_center = Dense(units=context_dims, activation="linear",
                        name="context_latent")(latent_coord)
    latent_coord = Multiply()([leap_center, context_inputs])
    for i, tau_dims in enumerate(reversed(block_dims)):
        latent_coord = Dense(units=tau_dims, activation="relu", name="dec_context_{}".format(i))(latent_coord)
    latent_coord = Dense(units=latent_dims, activation="linear",
                         name="latent_context")(latent_coord)
    new_coord = Add()([latent_coord, latent_inputs])

    return Model(inputs=block_inputs, outputs=new_coord, name=name)


def build_encoder_model(self, model_params, name= ""):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the encoder
    :return: a TF Model
    """

    x_inputs = Input(shape=(model_params.input_dims,), name="enc_inputs")
    c_inputs = []
    ensemble=[[] for i in range(model_params.nb_latent_components)]

    cond_enc_inputs_dims=[]

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="enc_cond_inputs_{}".format(i)))

    inputs = [x_inputs] + c_inputs

    if model_params.context_dims is not None:
        context_inputs = Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")
        inputs += [context_inputs]

    # Creation of the encoder block
    if len(c_inputs) >= 1 and 'encoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_enc_inputs = self.cond_embedding(c_inputs)
        else:
            if len(c_inputs) >= 2:
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
                                  kernel_initializer=model_params.kernel_initializer,
                                  bias_initializer=model_params.bias_initializer,
                                  name="encoder_block_{}".format(idx), activation="relu", 
                                  conditions_layer=model_params.cond_insert['encoder'],
                                  z_dim = model_params.latent_dims)
        enc_x = encoder_block(enc_inputs)

        for i in tf.range(0, model_params.nb_latent_components, 1):
            ensemble[i].append(Dense(units=model_params.latent_dims, activation='linear',
                                     kernel_initializer=model_params.kernel_initializer,
                                     bias_initializer=model_params.bias_initializer,
                                     name="latent_dense_{}_{}".format(idx,i+1))(enc_x))

    if model_params.nb_encoder_ensemble == 1:
        enc_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        enc_outputs = [average(ens_list) for ens_list in ensemble]

    if model_params.context_dims is not None:
        leap_block = build_leap_block(model_params.latent_dims, model_params.context_dims,
                                      model_params.leapae_dims)
        enc_outputs[0] = leap_block([enc_outputs[0], context_inputs])

    return Model(inputs=inputs, outputs=enc_outputs, name="encoder"+name)

def build_t2vencoder_model(self, model_params, name= ""):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the encoder
    :return: a TF Model
    """
    
    x_inputs = Input(shape=(model_params.t2v_dims+1,), name="enc_inputs")
        
    c_inputs = []
    ensemble=[[] for i in range(model_params.nb_latent_components)]

    cond_enc_inputs_dims=[]

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="enc_cond_inputs_{}".format(i)))

    inputs = [x_inputs] + c_inputs

    if model_params.context_dims is not None:
        context_inputs = Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")
        inputs += [context_inputs]

    # Creation of the encoder block
    if len(c_inputs) >= 1 and 'encoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_enc_inputs = self.cond_embedding(c_inputs)
        else:
            if len(c_inputs) >= 2:
                cond_enc_inputs = concatenate(c_inputs, name="concat_cond")
            else:
                cond_enc_inputs = c_inputs
        cond_enc_inputs_dims.append(K.int_shape(cond_enc_inputs[0])[-1])
        enc_inputs = [x_inputs, cond_enc_inputs]
    else:
        enc_inputs = [x_inputs]

    for idx in tf.range(0, model_params.nb_encoder_ensemble, 1):
        if model_params.with_Time2Vec:
            encoder_block = AE_blocks(input_dims=model_params.t2v_dims+1, cond_dims=cond_enc_inputs_dims,
                                  type=model_params.encoder_type, NN_dims=model_params.encoder_dims,
                                  kernel_initializer=model_params.kernel_initializer,
                                  bias_initializer=model_params.bias_initializer,
                                  name="encoder_block_{}".format(idx), activation="relu", 
                                  conditions_layer=model_params.cond_insert['encoder'],
                                  z_dim = model_params.latent_dims)
        enc_x = encoder_block(enc_inputs)

        for i in tf.range(0, model_params.nb_latent_components, 1):
            ensemble[i].append(Dense(units=model_params.latent_dims, activation='linear',
                                     kernel_initializer=model_params.kernel_initializer,
                                     bias_initializer=model_params.bias_initializer,
                                     name="latent_dense_{}_{}".format(idx,i+1))(enc_x))

    if model_params.nb_encoder_ensemble == 1:
        enc_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        enc_outputs = [average(ens_list) for ens_list in ensemble]

    if model_params.context_dims is not None:
        leap_block = build_leap_block(model_params.latent_dims, model_params.context_dims,
                                      model_params.leapae_dims)
        enc_outputs[0] = leap_block([enc_outputs[0], context_inputs])

    return Model(inputs=inputs, outputs=enc_outputs, name="encoder"+name)


def build_time2vec_model(self, model_params, name= ""):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the encoder
    :return: a TF Model
    """

    x_inputs = Input(shape=(model_params.input_dims,), name="enc_inputs")
    inputs = x_inputs 
    

    def prod(trans_inputs):
        return K.dot(trans_inputs[2],trans_inputs[1])

    def sinus(trans_inputs):
        
        inputs, prod, Phi, omega, phi = trans_inputs
        original = K.dot(inputs,omega) + phi
        trans = K.sin(prod + Phi)
        return trans, original

    #data = tf.keras.layers.Reshape((model_params.input_dims,1), name='data_reshape'+name)(inputs)
    Omega = tf.keras.layers.Dense(model_params.t2v_dims*model_params.input_dims)(inputs)
    Omega = tf.keras.layers.Reshape((model_params.t2v_dims,model_params.input_dims), name='Omega_reshape'+name)(Omega)
    omega = Dense(model_params.input_dims)(inputs)
    #omega = tf.keras.layers.Reshape((model_params.input_dims,1), name='omega_reshape'+name)(omega)
    Phi = Dense(model_params.t2v_dims)(inputs)
    #Phi = tf.keras.layers.Reshape((model_params.t2v_dims,1), name='Phi_reshape'+name)(Phi)
    phi = Dense(1)(inputs)

    prod = tf.keras.layers.Dot([2,1])([Omega, inputs])
    print(prod.shape, Phi.shape)
    prod = tf.keras.layers.Add()([prod, Phi])
    original = tf.keras.layers.Dot([1,1])([inputs,omega])
    original = tf.keras.layers.Add()([original, phi])
    #prod = tf.keras.layers.Reshape((model_params.t2v_dims,1))(prod)

    outputs = tf.keras.layers.Concatenate()([prod, original])
    print(outputs.shape)
    
    return Model(inputs=inputs, outputs=outputs, name="time2vec"+name)

def build_intel_latent_model(self, model_params, name= ""):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the encoder
    :return: a TF Model
    """

    z_inputs = Input(shape=(model_params.latent_dims,), name="enc_inputs")
    

    inputs = [z_inputs] 

    # Creation of the intel latent block
    
    encoder_block = AE_blocks(input_dims=model_params.latent_dims,
                                type='NNBlock_Unbiaised_model', NN_dims=model_params.latent_encoder_dims,
                                kernel_initializer=model_params.kernel_initializer,
                                bias_initializer=model_params.bias_initializer,
                                name="latent_encoder_block", activation="relu", 
                                z_dim = model_params.latent_dims)
    enc_z = encoder_block(inputs)

    binary = AE_blocks(input_dims=model_params.latent_dims,
                                type='NNBlock_binary_model', NN_dims=model_params.latent_encoder_dims,
                                kernel_initializer=model_params.kernel_initializer,
                                bias_initializer=model_params.bias_initializer,
                                name="binary_block", activation="relu",
                                z_dim = model_params.latent_dims)

    z_hat = binary(inputs)

    return Model(inputs=z_inputs, outputs=[enc_z, z_hat], name="latent"+name)
    
def build_encoder_oldmodel(self, model_params, name= ""):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the encoder
    :return: a TF Model
    """
    x_inputs = Input(shape=(model_params.input_dims,), name='enc_x_true')

    c_inputs = []
    ensemble=[[] for i in range(model_params.nb_latent_components)]

    cond_enc_inputs_dims=[]
    c_inputs.append(Input(shape=(model_params.cond_dims,), name="enc_cond_inputs"))

    inputs = [x_inputs] + c_inputs

    if model_params.context_dims is not None:
        context_inputs = Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")
        inputs += [context_inputs]

    # Creation of the encoder block
    if len(c_inputs) >= 1 and 'encoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_enc_inputs = self.cond_embedding(c_inputs)
        else:
            if len(c_inputs) >= 2:
                cond_enc_inputs = concatenate(c_inputs, name="concat_cond")
            else:
                cond_enc_inputs = c_inputs
        cond_enc_inputs_dims.append(K.int_shape(cond_enc_inputs[0])[-1])
        enc_inputs = [x_inputs, cond_enc_inputs]
    else:
        enc_inputs = [x_inputs]

    if(self.cond_dim>=1):

        cond_inputs = Input(shape=(self.cond_dim,), name='enc_cond')
        x = concatenate([x_inputs, cond_inputs], name='enc_input')
    else:
        cond_inputs = Input(shape=(0,), name='enc_cond')
        x = concatenate([x_inputs, cond_inputs], name='enc_input')

    nLayers = len(model_params.encoder_dims)
    for idx, layer_dim in enumerate(model_params.encoder_dims):
        if (idx<(nLayers-1)):
            x = concatenate([Dense(units=layer_dim, activation='relu')(x),cond_inputs], name="enc_dense_{}".format(idx))
        else:
            x = Dense(units=layer_dim, activation='relu', name="enc_dense_{}".format(idx))(x)
   
    z_mu = Dense(units=self.z_dim, activation='linear', name="latent_dense_mu")(x)
    z_log_sigma = Dense(units=self.z_dim, activation='linear', name='latent_dense_log_sigma')(x)

    return Model(inputs=[x_inputs, cond_inputs], outputs=[z_mu, z_log_sigma], name='encoder'+name)

    x_inputs = Input(shape=(model_params.input_dims,), name="enc_inputs")
    c_inputs = []
    ensemble=[[] for i in range(model_params.nb_latent_components)]

    cond_enc_inputs_dims=[]
    c_inputs.append(Input(shape=(model_params.cond_dims,), name="enc_cond_inputs"))

    inputs = [x_inputs] + c_inputs

    if model_params.context_dims is not None:
        context_inputs = Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")
        inputs += [context_inputs]

    # Creation of the encoder block
    if len(c_inputs) >= 1 and 'encoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_enc_inputs = self.cond_embedding(c_inputs)
        else:
            if len(c_inputs) >= 2:
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
                                  kernel_initializer=model_params.kernel_initializer,
                                  bias_initializer=model_params.bias_initializer,
                                  name="encoder_block_{}".format(idx), activation="relu", 
                                  conditions_layer=model_params.cond_insert['encoder'],
                                  z_dim = model_params.latent_dims)
        enc_x = encoder_block(enc_inputs)

        for i in tf.range(0, model_params.nb_latent_components, 1):
            ensemble[i].append(Dense(units=model_params.latent_dims, activation='linear',
                                     kernel_initializer=model_params.kernel_initializer,
                                     bias_initializer=model_params.bias_initializer,
                                     name="latent_dense_{}_{}".format(idx,i+1))(enc_x))

    if model_params.nb_encoder_ensemble == 1:
        enc_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        enc_outputs = [average(ens_list) for ens_list in ensemble]

    if model_params.context_dims is not None:
        leap_block = build_leap_block(model_params.latent_dims, model_params.context_dims,
                                      model_params.leapae_dims)
        enc_outputs[0] = leap_block([enc_outputs[0], context_inputs])

    return Model(inputs=inputs, outputs=enc_outputs, name="encoder"+name)


def build_decoder_model(self, model_params):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the decoder
    :return: a TF Model
    """

    encoded_inputs = Input(shape=(model_params.latent_dims,), name="encoded_inputs")
    c_inputs = []
    ensemble = [[] for i in range(model_params.nb_decoder_outputs)]
    cond_dec_inputs_dims = []

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="dec_cond_inputs_{}".format(i)))

    inputs = [encoded_inputs] + c_inputs

    if len(c_inputs)>=1 and 'decoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_dec_inputs = self.cond_embedding(c_inputs)
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
                                  kernel_initializer=model_params.kernel_initializer,
                                  bias_initializer=model_params.bias_initializer,
                                  name="decoder_block_{}".format(idx), activation="relu",
                                  conditions_layer=model_params.cond_insert['decoder'])

        dec_x = decoder_block(dec_inputs)

        for i in tf.range(0, model_params.nb_decoder_outputs, 1):
            ensemble[i].append(Dense(units=model_params.output_dims, activation='linear',
                                     kernel_initializer=model_params.kernel_initializer,
                                     bias_initializer=model_params.bias_initializer,
                                     name="dec_output_{}_{}".format(idx,i+1),)(dec_x))

    if model_params.nb_decoder_ensemble == 1:
        dec_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        dec_outputs = [average(ens_list) for ens_list in ensemble]

    return Model(inputs=inputs, outputs=dec_outputs, name="decoder")

def build_decoder_oldmodel(self, model_params):
    """

    :param self: self of the CVAE Class
    :param model_params: ModelParams class, with instances gathering the parameters of each layer of the decoder
    :return: a TF Model
    """

    encoded_inputs = Input(shape=(model_params.latent_dims,), name="encoded_inputs")
    c_inputs = []
    ensemble = [[] for i in range(model_params.nb_decoder_outputs)]
    cond_dec_inputs_dims = []

    c_inputs.append(Input(shape=(model_params.cond_dims,), name="dec_cond_inputs"))

    inputs = [encoded_inputs] + c_inputs

    if len(c_inputs)>=1 and 'decoder' in model_params.cond_insert.keys():
        if model_params.with_embedding:
            cond_dec_inputs = self.cond_embedding(c_inputs)
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
                                  kernel_initializer=model_params.kernel_initializer,
                                  bias_initializer=model_params.bias_initializer,
                                  name="decoder_block_{}".format(idx), activation="relu",
                                  conditions_layer=model_params.cond_insert['decoder'])

        dec_x = decoder_block(dec_inputs)

        for i in tf.range(0, model_params.nb_decoder_outputs, 1):
            ensemble[i].append(Dense(units=model_params.output_dims, activation='linear',
                                     kernel_initializer=model_params.kernel_initializer,
                                     bias_initializer=model_params.bias_initializer,
                                     name="dec_output_{}_{}".format(idx,i+1),)(dec_x))

    if model_params.nb_decoder_ensemble == 1:
        dec_outputs = [ens_list[0] for ens_list in ensemble]
    else:
        dec_outputs = [average(ens_list) for ens_list in ensemble]

    return Model(inputs=inputs, outputs=dec_outputs, name="decoder")

def build_guidedencoder_model(self, model_params, x_encoder, list_condencoder):
    c_inputs = []
    cx_inputs = []
    for i, input_dim in enumerate(model_params.input_dims):
        if i == 0:
            x_inputs = [Input(shape=(input_dim,), name=f"inputs_{i}")]
        else:
            cx_inputs.append(Input(shape=(input_dim,), name=f"inputs_{i}"))

    for i, c_dims in enumerate(model_params.cond_dims):
        c_inputs.append(Input(shape=(c_dims,), name="enc_cond_inputs_{}".format(i)))

    inputs = x_inputs + cx_inputs + c_inputs
    enc_inputs = x_inputs + c_inputs

    if model_params.context_dims is not None:
        context_inputs = Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")
        inputs += [context_inputs]
        enc_inputs += [context_inputs]

    list_latent_cond = []

    x_encoded = x_encoder(enc_inputs)
    x_encoded[0] = Lambda(lambda z: z*0, name="neutralized_mu_encoded")(x_encoded[0])

    for i, c_encoder in enumerate(list_condencoder):
        cond_nb = Lambda(lambda x: tf.one_hot(tf.constant(i,dtype='int32',shape=(1,)),
                                              len(list_condencoder)), name=f"latent_dense_mu_c_class_{i}")(cx_inputs[i])

        c_encoded = c_encoder(cx_inputs[i])
        c_latent_dim = Dense(units=model_params.latent_dims, activation='hard_sigmoid',
                             name= "latent_dense_mu_c_dim_{}".format(i))(cond_nb)

        list_latent_cond.append(Multiply()([c_encoded, c_latent_dim]))

    x_encoded[0] = Add()([x_encoded[0]]+list_latent_cond)

    return Model(inputs=inputs, outputs=x_encoded, name="guided_encoder")

