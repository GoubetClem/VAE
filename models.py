from abc import ABC, abstractmethod
import os
import pickle
import copy
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda

from AE_blocks import *


class AE_Model(ABC):

    def __init__(self, VAE_params):
        """

        """
        assert ("name" in VAE_params.__dict__.keys())
        assert ("folder" in VAE_params.__dict__.keys())

        self.VAE_params = VAE_params
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
            out_dir = self.VAE_params.folder

        folder = os.path.join(out_dir, "model")
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s.hdf5" %(block))
            getattr(self, block).save(filepath = filepath)

        graph_params = copy.deepcopy(self.VAE_params.model_params.__dict__)
        with open(self.VAE_params.name+'_model_architecture.obj', 'w') as config_model_file:
            pickle.dump(graph_params, config_model_file)

    def load_weights(self, out_dir = None, retrieve_model_architecture=True):
        if out_dir is None:
            out_dir = self.VAE_params.folder

        folder = os.path.join(out_dir, "model")

        if retrieve_model_architecture:
            with open(self.VAE_params.name + '_model_architecture.obj', 'r') as config_model_file:
                self.VAE_params.model_params.__dict__ = pickle.load(config_model_file)

            self.VAE_params.set_training_params()

            self.build_model(self, self.VAE_params)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s.hdf5" %(block))
            getattr(self, block).load_weights(filepath = filepath)


class CVAE(AE_Model):

    def __init__(self, VAE_params):
        AE_Model.__init__(self, VAE_params = VAE_params)

    def build_model(self, VAE_params, custom_encoder_model=None, custom_decoder_model=None):
        self.VAE_params.model_params = VAE_params.model_params

        # getting the graph inputs
        x_inputs = Input(shape=(self.VAE_params.model_params.input_dims,), name="x_inputs")
        c_inputs = []

        for i, c_dims in enumerate(self.VAE_params.model_params.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        inputs = [x_inputs] + c_inputs

        # Setting the AE architecture
        if custom_encoder_model is not None:
            self.encoder = custom_encoder_model
        else:
            self.encoder = build_encoder_model(self, model_params=self.VAE_params.model_params)

        if custom_decoder_model is not None:
            self.decoder = custom_decoder_model
        else:
            self.decoder = build_decoder_model(self, model_params=self.VAE_params.model_params)

        # Model AE graph
        enc_outputs = self.encoder(inputs)

        if self.VAE_params.model_params.nb_latent_components ==1:
            dec_inputs = enc_outputs + c_inputs
        else:
            z = Lambda(self.VAE_params.model_params.reparametrize, name="reparametrizing_layer")(enc_outputs)
            dec_inputs = [z] + c_inputs

        dec_outputs = self.decoder(dec_inputs)
        x_hat = dec_outputs[0]

        self.model = Model(inputs=inputs, outputs=x_hat, name="cae")
        self.model.summary()
        self.blocks.append("model")

        if self.VAE_params.model_params.with_embedding:
            self.to_embedding.summary()
            self.blocks.append("to_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

        # Training objectives settings
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)

        if self.VAE_params.model_params.with_embedding:
            emb_outputs = self.to_embedding(inputs)
            model_loss = self.VAE_params.training_params.loss._get_loss_function(latent_components=enc_outputs,
                                                                             latent_sampling=dec_inputs[0],
                                                                             cond_true=c_inputs,
                                                                                 dec_outputs = dec_outputs,
                                                                             embedding_outputs = emb_outputs)
        else:
            model_loss = self.VAE_params.training_params.loss._get_loss_function(latent_components=enc_outputs,
                                                                            latent_sampling=dec_inputs[0],
                                                                            cond_true=c_inputs,
                                                                                 dec_outputs = dec_outputs)

        print("Losses and associated weight involved in the model: ")
        [print(loss_key, " : ",
               self.VAE_params.training_params.loss.loss_weights[loss_key]) for loss_key in self.VAE_params.training_params.loss.losses.keys()]

        self.model.compile(loss=model_loss, optimizer=optimizer, experimental_run_tf_function=False)
















"""


class old_CAE(AE_Model):

    def __init__(self, cond_dims=[], with_embedding=False, **kwargs):
        AE_Model.__init__(self, **kwargs)
        self.cond_dims = cond_dims
        self.with_embedding = with_embedding

        if with_embedding and "emb_dims" not in kwargs:
            raise Exception('Please specify embeddings layers dimensions!')


        if with_embedding:
            assert len(kwargs["emb_dims"]) == len(cond_dims) + 1
            self.emb_dims = kwargs["emb_dims"]


    def build_model(self, input_dims, latent_dims, encoder_dims=[24], encoder_type="NNBlockCond_model",
                    decoder_dims=[24], decoder_type="InceptionBlock_model", **kwargs):

        #getting the inputs
        x_inputs = Input(shape=(input_dims,), name="x_inputs")
        c_inputs = []

        for i, c_dims in enumerate(self.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        inputs = [x_inputs] + c_inputs

        # Creation of the encoder block
        self.encoder = encoder_model(self, type=encoder_type, input_dims=input_dims, latent_dims=latent_dims,
                                     encoder_dims=encoder_dims, number_outputs=1)

        self.decoder = decoder_model(self, type=decoder_type, input_dims=input_dims, latent_dims=latent_dims,
                                     decoder_dims=decoder_dims)

        #Model AE settings
        enc_outputs = self.encoder(inputs)

        dec_inputs = [enc_outputs] + c_inputs

        x_hat = self.decoder(dec_inputs)
        self.model = Model(inputs=inputs, outputs=x_hat, name="cae")
        self.model.summary()
        self.blocks.append("model")

        if self.with_embedding:
            self.to_embedding.summary()
            self.blocks.append("to_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

    def build_objectives(self, lr=3e-4, recon_loss="mae"):

        optimizer = optimizers.Adam(lr=lr)

        self.model.compile(loss=recon_loss, optimizer=optimizer, experimental_run_tf_function=False)


class old_CVAE(old_CAE):

    def build_model(self, input_dims, latent_dims, encoder_dims=[24], encoder_type="NNBlockCond_model",
                    decoder_dims=[24], decoder_type="InceptionBlock_model", **kwargs):

        # getting the inputs
        x_inputs = Input(shape=(input_dims,), name="x_inputs")
        c_inputs = []

        for i, c_dims in enumerate(self.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        inputs = [x_inputs] + c_inputs

        # Creation of the encoder block
        self.encoder = encoder_model(self, type=encoder_type, input_dims=input_dims, latent_dims=latent_dims,
                                     encoder_dims=encoder_dims, number_outputs=2)

        self.decoder = decoder_model(self, type=decoder_type, input_dims=input_dims, latent_dims=latent_dims,
                                     decoder_dims=decoder_dims)

        #Model AE settings
        z_mu, z_log_sigma = self.encoder(inputs)

        z = Lambda(GaussianSampling, name="reparametrizing_layer")([z_mu, z_log_sigma])

        dec_inputs = [z] + c_inputs

        x_hat = self.decoder(dec_inputs)
        self.model = Model(inputs=inputs, outputs=x_hat, name="cvae")
        self.model.summary()
        self.blocks.append("model")

        if self.with_embedding:
            self.to_embedding.summary()
            self.blocks.append("to_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

    def build_objectives(self, lr=3e-4, recon_loss="mae", loss_weights={"recon_loss" : 1.}, **kwargs):

        optimizer = optimizers.Adam(lr=lr)
        self.vae_loss = VAE_Loss(recon_loss=recon_loss, loss_weights=loss_weights, **kwargs)

        model_loss = self.vae_loss(latent_mu = self.model.get_layer("latent_dense_0").output,
                                   latent_log_sigma=self.model.get_layer("latent_dense_1").output,
                                   latent_sampling=self.model.get_layer("reparametrizing_layer").output,
                                   **kwargs)

        self.model.compile(loss=model_loss, optimizer=optimizer, experimental_run_tf_function=False)



"""
