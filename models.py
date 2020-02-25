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
        self.encoder = encoder_model(self, type="NNBlockCond_model", input_dims=input_dims, latent_dims=latent_dims,
                                     encoder_dims=encoder_dims, number_outputs=1)

        self.decoder = decoder_model(self, type="InceptionBlock_model", input_dims=input_dims, latent_dims=latent_dims,
                                     decoder_dims=decoder_dims)



        #Model AE settings
        enc_outputs = self.encoder(inputs)

        dec_inputs = [enc_outputs] + c_inputs

        x_hat = self.decoder(dec_inputs)
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

