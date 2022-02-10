from abc import ABC, abstractmethod
import os
import json
import copy
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda
from src.AE_blocks import *


class GAN_Model(ABC):

    def __init__(self, GAN_params):
        """

        """
        assert ("name" in GAN_params.__dict__.keys())
        assert ("folder" in GAN_params.__dict__.keys())

        self.GAN_params = GAN_params
        self.model = None
        self.blocks=[]


    def maketrainable(self, modelpart=['generator'], boolean=True):
        for mpart in modelpart:
            print("Change trainable status of {} layers".format(mpart))
            assert(mpart in self.__dict__.keys())

            input_names =  getattr(self, mpart).input_names
            getattr(self, mpart).trainable = boolean
            for layer in getattr(self, mpart).layers:
                if (layer.name not in input_names):
                    layer.trainable = boolean

        if mpart not in ["generator", "discriminator"]:
            for block in ["generator", "discriminator"]:
                submodel_name = getattr(self, mpart).name
                if submodel_name in [lay.name for lay in getattr(self, block).layers]:
                    input_names = getattr(self, block).get_layer(submodel_name).input_names
                    getattr(self, block).get_layer(submodel_name).trainable = boolean
                    for layer in getattr(self, block).get_layer(submodel_name).layers:
                        if (layer.name not in input_names):
                            layer.trainable = boolean

        self.save()
        optimizer = self.GAN_params.training_params.optimizer(self.GAN_params.training_params.lr)
        self.model.compile(loss=self.GAN_params.training_params.loss.losses["recon_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)

        self.load_model(retrieve_model_architecture=False)


    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    def train(self, blocks_totrain = ["generator", "discriminator"], *args,  **kwargs):

        print("## START TRAINING ##")

        if "validation_split" not in kwargs:
            kwargs["validation_split"] = 0.1

        training_history = self.model.fit(*args, **kwargs)

        print("## END TRAINING ##")

        plt.plot(training_history.history['loss'])
        plt.plot(training_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        self.save()
        print("## MODEL SAVED ##")

    def save(self, out_dir=None):
        if out_dir is None:
            out_dir = self.GAN_params.folder

        folder = os.path.join(out_dir, "model")
        if not os.path.isdir(folder):
            os.makedirs(folder)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s" %(block))
            getattr(self, block).save_weights(filepath = filepath, save_format="tf")

        graph_params = copy.deepcopy(self.GAN_params.model_params.__dict__)
        filename = os.path.join(folder, self.GAN_params.name+'_model_architecture.json')
        with open(filename, 'w') as config_model_file:
            json.dump(graph_params, config_model_file, indent=4)

    def load_model(self, out_dir = None, retrieve_model_architecture=True, training_params=None):
        if out_dir is None:
            out_dir = self.GAN_params.folder

        folder = os.path.join(out_dir, "model")

        if retrieve_model_architecture:
            filename = os.path.join(folder, self.GAN_params.name + '_model_architecture.json')
            with open(filename, 'r') as config_model_file:
                model_params_dict = json.load(config_model_file)

            for k,v in model_params_dict.items():
                setattr(self.GAN_params.model_params, k, v)

            self.GAN_params.set_training_params()
            if training_params is not None:
                self.GAN_params.training_params = training_params

            self.build_model(self.GAN_params)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s" %(block))
            getattr(self, block).load_weights(filepath = filepath)