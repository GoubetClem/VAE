import os
from tensorflow.keras.optimizers import Adam
from reparametrize_functions import *
from loss_class import *

class ModelParams():
    def __init__(self):
        self.cond_dims = []
        self.with_embedding = False
        self.emb_dims = []
        self.reparametrize = GaussianSampling

        self.input_dims=48
        self.latent_dims = 4
        self.nb_latent_components = 2
        self.nb_encoder_ensemble = 1
        self.encoder_dims = [48,48,24,12]
        self.encoder_type = "NNBlockCond_model"
        self.nb_decoder_outputs = 1
        self.nb_decoder_ensemble = 1
        self.decoder_dims = [48,48,24,12]
        self.decoder_type = "InceptionBlock_model"


class TrainingParams():
    def __init__(self):
        self.optimizer=Adam
        self.lr=3e-4
        self.loss = VAELoss()


class VAE_params():

    def __init__(self, **kwargs):

        if "name" not in kwargs:
            raise Exception('Please specify model name!')

        self.name = kwargs["name"]

        if "out_dir" not in kwargs:
            raise Exception('Please specify model savings folder path')

        self.folder = os.path.join(kwargs["out_dir"], self.name)
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)

    def set_model_params(self):
        self.model_params = ModelParams()

        print("Default values for model architecture are:")
        for key, value in self.model_params.__dict__.items():
            print(key, " := ", value)

    def set_training_params(self):
        self.training_params = TrainingParams()
        if "lr" in self.training_params.__dict__.keys():
            print(
                "Default parameters for training are a L1 loss with Adam optimizer with a learning rate of %1s" % self.training_params.lr)

