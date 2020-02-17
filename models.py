from abc import ABC, abstractmethod
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Model, Input

from tensorflow.keras import optimizers

from AE_blocks import *

class AE_Model(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """

        """

        if "name" not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs["name"]

        if "out_dir" not in kwargs:
            raise Exception('Please specify model savings folder path')

        self.folder = os.path.join(kwargs["out_dir"], self.name)
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

        self.model = None
        self.blocks=[]

    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    def train(self, *args,  **kwargs):

        print("## START TRAINING ##")

        training_history = self.model.fit(validation_split=0.1, *args, **kwargs)

        print("## END TRAINING ##")

        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        self.save()
        print("## MODEL SAVED ##")

    def save(self, out_dir=None):
        if out_dir is None:
            out_dir = self.folder
            out_dir = self.folder

        folder = os.path.join(out_dir)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s.hdf5" %(block))
            getattr(self, block).save(filepath = filepath)

    def load_weights(self, out_dir = None):
        if out_dir is None:
            out_dir = self.folder

        folder = os.path.join(out_dir)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s.hdf5" %(block))
            getattr(self, block).load_weights(filepath = filepath)

class CAE(AE_Model):

    def __init__(self, cond_dims=[], with_embedding=False, is_L2_Loss=True, lr=0.001, **kwargs):
        AE_Model.__init__(self, **kwargs)
        self.cond_dims = cond_dims
        self.with_embedding = with_embedding
        self.is_L2_Loss = is_L2_Loss
        self.lr = lr

        if with_embedding and "emb_dims" not in kwargs:
            raise Exception('Please specify embeddings layers dimensions!')

        if with_embedding:
            self.emb_dims = kwargs["emb_dims"]

    def build_model(self, input_dims, latent_dims, encoder_dims=[24], decoder_dims=[24], **kwargs):
        """

        :param input_dims:
        :param latent_dims:
        :param encoder_dims:
        :param decoder_dims:
        :param kwargs:
        :return:
        """

        #getting the inputs
        x_inputs = Input(shape=(input_dims,), name="x_inputs")
        c_inputs = []

        for i, c_dims in enumerate(self.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        inputs = [x_inputs] + c_inputs

        # Creation of the encoder block
        if len(c_inputs)==0:
            encoder_block = NNBlock_model(input_dims, NN_dims=encoder_dims, name="encoder_block", activation="relu")
            enc_x = encoder_block(x_inputs)
        else:
            if self.with_embedding:
                self.to_embedding=  EmbeddingBlock_model(self.cond_dims, self.emb_dims, latent_dims, activation="relu", name="emb", has_BN=True)
                cond_enc_inputs = self.to_embedding(c_inputs)
            else:
                cond_enc_inputs = concatenate(c_inputs, name="concat_cond")

            encoder_block = NNBlockCond_model(input_dims, self.cond_dims, NN_dims=encoder_dims, name="encoder_block", activation="relu")
            enc_x = encoder_block([x_inputs, cond_enc_inputs])

        enc_outputs = Dense(units=latent_dims, activation='linear', name="latent_dense_mu")(enc_x)

        self.encoder = Model(inputs=inputs, outputs=enc_outputs, name="encoder")

        # creation of the decoder block
        dec_inputs = Input(shape=(latent_dims,), name="dec_inputs")
        decoder_block = InceptionBlock_model(latent_dims, NN_dims=decoder_dims, name="decoder_block", activation="relu")
        dec_x = decoder_block(dec_inputs)
        dec_outputs = Dense(input_dims, activation='linear', name='dec_output')(dec_x)
        self.decoder = Model(inputs=dec_inputs, outputs=dec_outputs, name="decoder")

        #Model AE settings
        x_hat = self.decoder(enc_outputs)
        self.model = Model(inputs=inputs, outputs=x_hat, name="cae")
        self.model.summary()
        self.blocks.append("model")

        optimizer = optimizers.Adam(lr=self.lr)
        if self.is_L2_Loss:
            loss="mse"
        else:
            loss = "mae"
        self.model.compile(loss=loss, optimizer=optimizer, experimental_run_tf_function=False)

        #Blocks callers

        if self.with_embedding:
            self.to_embedding.summary()
            self.blocks.append("to_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

