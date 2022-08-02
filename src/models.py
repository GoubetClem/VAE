from abc import ABC, abstractmethod
import os
import json
import copy
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Lambda, Layer, Dropout

from src.reparametrize_functions import *
from src.AE_blocks import *
from src.AE_blocks import build_decoder_oldmodel, build_encoder_oldmodel, build_t2vencoder_model

class AE_Model(ABC):

    def __init__(self, VAE_params):
        """

        """
        assert ("name" in VAE_params.__dict__.keys())
        assert ("folder" in VAE_params.__dict__.keys())

        self.VAE_params = VAE_params
        self.model = None
        self.blocks=[]


    def maketrainable(self, modelpart=['encoder'], boolean=True):
        for mpart in modelpart:
            print("Change trainable status of {} layers".format(mpart))
            assert(mpart in self.__dict__.keys())

            input_names =  getattr(self, mpart).input_names
            getattr(self, mpart).trainable = boolean
            for layer in getattr(self, mpart).layers:
                if (layer.name not in input_names):
                    layer.trainable = boolean

        if mpart not in ["encoder", "decoder"]:
            for block in ["encoder", "decoder"]:
                submodel_name = getattr(self, mpart).name
                if submodel_name in [lay.name for lay in getattr(self, block).layers]:
                    input_names = getattr(self, block).get_layer(submodel_name).input_names
                    getattr(self, block).get_layer(submodel_name).trainable = boolean
                    for layer in getattr(self, block).get_layer(submodel_name).layers:
                        if (layer.name not in input_names):
                            layer.trainable = boolean

        #self.save()
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)
        if not self.VAE_params.model_params.is_oldCVAE:
            self.model.compile(loss=self.VAE_params.training_params.loss.losses["recon_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)

        self.load_model(retrieve_model_architecture=False)


    @abstractmethod
    def build_model(self, *args, **kwargs):
        pass

    def train(self, *args,  **kwargs):

        print("## START TRAINING ##")

        if "validation_split" not in kwargs:
            kwargs["validation_split"] = 0.1

        self.training_history = self.model.fit(*args, **kwargs)

        print("## END TRAINING ##")

        plt.plot(self.training_history.history['loss'])
        plt.plot(self.training_history.history['val_loss'])
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
            out_dir = self.VAE_params.folder

        folder = os.path.join(out_dir, "model")
        if not os.path.isdir(folder):
            os.makedirs(folder)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s" %(block))
            getattr(self, block).save_weights(filepath = filepath, save_format="tf")

        graph_params = copy.deepcopy(self.VAE_params.model_params.__dict__)
        filename = os.path.join(folder, self.VAE_params.name+'_model_architecture.json')
        with open(filename, 'w') as config_model_file:
            json.dump(graph_params, config_model_file, indent=4)

    def load_model(self, out_dir = None, retrieve_model_architecture=True, training_params=None):
        if out_dir is None:
            out_dir = self.VAE_params.folder

        folder = os.path.join(out_dir, "model")

        if retrieve_model_architecture:
            filename = os.path.join(folder, self.VAE_params.name + '_model_architecture.json')
            with open(filename, 'r') as config_model_file:
                model_params_dict = json.load(config_model_file)

            for k,v in model_params_dict.items():
                setattr(self.VAE_params.model_params, k, v)

            self.VAE_params.set_training_params()
            if training_params is not None:
                self.VAE_params.training_params = training_params

            self.build_model(self.VAE_params)

        for block in self.blocks:
            filepath = os.path.join(folder, "%s" %(block))
            getattr(self, block).load_weights(filepath = filepath)

class CVAEpond(AE_Model):

    def __init__(self, VAE_params):
        AE_Model.__init__(self, VAE_params = VAE_params)

    def build_model(self, VAE_params, custom_encoder_model=None, custom_decoder_model=None, custom_timetovec_model = None):
        """

        :param VAE_params: VAE_params class, subclasses of parameters needed to build each layers of the autoencoders and training paramaters to be used in the compile function
        :param custom_encoder_model: TF Model, to be used as encoder model in the graph
        :param custom_decoder_model: TF Model, to be used as decoder model in the graph
        :return: build graph and compile model in the CVAE Class
        """
        self.VAE_params.model_params = VAE_params.model_params
        if self.VAE_params.model_params.with_embedding :
            assert(len(self.VAE_params.model_params.cond_dims) + 1 == len(self.VAE_params.model_params.emb_dims))

        # getting the graph inputs
        x_inputs = Input(shape=(self.VAE_params.model_params.input_dims,), name="x_inputs")
        c_inputs = []
        context_inputs = []

        for i, c_dims in enumerate(self.VAE_params.model_params.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        if self.VAE_params.model_params.context_dims is not None:
            context_inputs = [Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")]
        
        inputs = [x_inputs] + c_inputs + context_inputs

        # Setting the AE architecture
        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding = build_embedding_model(self, model_params=self.VAE_params.model_params)

        if self.VAE_params.model_params.with_Time2Vec and custom_timetovec_model is not None: 
            self.timetovec_model = custom_timetovec_model

        elif self.VAE_params.model_params.with_Time2Vec and custom_timetovec_model is None: 
            self.timetovec_model = build_time2vec_model(self, model_params=self.VAE_params.model_params)
        
        if custom_encoder_model is not None:
            self.encoder = custom_encoder_model
        elif self.VAE_params.model_params.with_Time2Vec :
            self.encoder = build_t2vencoder_model(self, model_params=self.VAE_params.model_params)
        else:
            self.encoder = build_encoder_model(self, model_params=self.VAE_params.model_params)

        if custom_decoder_model is not None:
            self.decoder = custom_decoder_model
        else:
            self.decoder = build_decoder_model(self, model_params=self.VAE_params.model_params)

        # Model AE graph
        #time2vec
        if self.VAE_params.model_params.with_Time2Vec:
            t2v_outputs = self.timetovec_model(inputs)
            #encoding
            enc_outputs = self.encoder([t2v_outputs])
        else:    
            enc_outputs = self.encoder([inputs])
            
        if (self.VAE_params.model_params.nb_latent_components == 1):
            dec_inputs = [enc_outputs] + c_inputs
        else:
            z = Lambda(eval(self.VAE_params.model_params.reparametrize), name="reparametrizing_layer")(enc_outputs)
            dec_inputs = [z] + c_inputs

        #decoding
        dec_outputs = self.decoder(dec_inputs)

        if self.VAE_params.model_params.nb_decoder_outputs == 1:
            x_hat = dec_outputs
        else:
            x_hat = Lambda(eval(self.VAE_params.model_params.reparametrize), name="loglikelihood_layer")(dec_outputs)

        self.model = Model(inputs=inputs, outputs=x_hat, name="cvae")

        vae_args = dict(
            latent_components=enc_outputs,
            latent_sampling=dec_inputs[0],
            cond_true=c_inputs,
            y_true = x_inputs,
            y_pred = x_hat,
            dec_outputs = dec_outputs
        )
        if self.VAE_params.model_params.with_embedding:
            vae_args.update(dict(embedding_outputs=self.cond_embedding(c_inputs)))


        self.model.summary()
        self.blocks.append("model")

        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding.summary()
            self.blocks.append("cond_embedding")

        if self.VAE_params.model_params.with_Time2Vec:
            self.timetovec_model.summary()
            self.blocks.append("timetovec_model")

        self.encoder.summary()
        self.blocks.append("encoder")
        
        self.decoder.summary()
        self.blocks.append("decoder")

        # Training objectives settings
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)
        if len(list(self.VAE_params.training_params.loss.loss_weights.keys()))> 1:
            pond_loss = self.VAE_params.training_params.loss._get_loss_function_with_key("pond_loss",**vae_args)
            self.model.add_loss(self.VAE_params.training_params.loss._get_loss_function_pond(**vae_args))
            #self.model.add_loss(pond_loss)
            self.model.add_metric(pond_loss,aggregation='mean', name="pond_loss")
            for key in self.VAE_params.training_params.loss.loss_weights.keys():
                if key != "pond_loss":
                    metric = self.VAE_params.training_params.loss._get_loss_function_with_key(key,**vae_args)
                    self.model.add_loss(metric)
                    self.model.add_metric(metric,aggregation='mean', name=key)

        print("Losses and associated weight involved in the model: ")
        [print(loss_key, " : ",
               self.VAE_params.training_params.loss.loss_weights[loss_key]) for loss_key in self.VAE_params.training_params.loss.losses.keys()]

        self.model.compile(loss=self.VAE_params.training_params.loss.losses["pond_loss"],
                           loss_weights=self.VAE_params.training_params.loss.loss_weights["pond_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)

class CVAE(AE_Model):

    def __init__(self, VAE_params):
        AE_Model.__init__(self, VAE_params = VAE_params)

    def build_model(self, VAE_params, custom_encoder_model=None, custom_decoder_model=None, custom_timetovec_model = None):
        """

        :param VAE_params: VAE_params class, subclasses of parameters needed to build each layers of the autoencoders and training paramaters to be used in the compile function
        :param custom_encoder_model: TF Model, to be used as encoder model in the graph
        :param custom_decoder_model: TF Model, to be used as decoder model in the graph
        :return: build graph and compile model in the CVAE Class
        """
        self.VAE_params.model_params = VAE_params.model_params
        if self.VAE_params.model_params.with_embedding :
            assert(len(self.VAE_params.model_params.cond_dims) + 1 == len(self.VAE_params.model_params.emb_dims))

        # getting the graph inputs
        x_inputs = Input(shape=(self.VAE_params.model_params.input_dims,), name="x_inputs")
        c_inputs = []
        context_inputs = []

        for i, c_dims in enumerate(self.VAE_params.model_params.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        if self.VAE_params.model_params.context_dims is not None:
            context_inputs = [Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")]
        
        inputs = [x_inputs] + c_inputs + context_inputs

        # Setting the AE architecture
        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding = build_embedding_model(self, model_params=self.VAE_params.model_params)

        if self.VAE_params.model_params.with_Time2Vec and custom_timetovec_model is not None: 
            self.timetovec_model = custom_timetovec_model

        elif self.VAE_params.model_params.with_Time2Vec and custom_timetovec_model is None: 
            self.timetovec_model = build_time2vec_model(self, model_params=self.VAE_params.model_params)
        
        if custom_encoder_model is not None:
            self.encoder = custom_encoder_model
        elif self.VAE_params.model_params.with_Time2Vec :
            self.encoder = build_t2vencoder_model(self, model_params=self.VAE_params.model_params)
        else:
            self.encoder = build_encoder_model(self, model_params=self.VAE_params.model_params)

        if custom_decoder_model is not None:
            self.decoder = custom_decoder_model
        else:
            self.decoder = build_decoder_model(self, model_params=self.VAE_params.model_params)

        # Model AE graph
        #time2vec
        if self.VAE_params.model_params.with_Time2Vec:
            t2v_outputs = self.timetovec_model(inputs)
            #encoding
            enc_outputs = self.encoder([t2v_outputs])
        else:    
            enc_outputs = self.encoder([inputs])
            
        if (self.VAE_params.model_params.nb_latent_components == 1):
            dec_inputs = [enc_outputs] + c_inputs
        else:
            z = Lambda(eval(self.VAE_params.model_params.reparametrize), name="reparametrizing_layer")(enc_outputs)
            dec_inputs = [z] + c_inputs

        #decoding
        dec_outputs = self.decoder(dec_inputs)

        if self.VAE_params.model_params.nb_decoder_outputs == 1:
            x_hat = dec_outputs
        else:
            x_hat = Lambda(eval(self.VAE_params.model_params.reparametrize), name="loglikelihood_layer")(dec_outputs)

        self.model = Model(inputs=inputs, outputs=x_hat, name="cvae")

        vae_args = dict(
            latent_components=enc_outputs,
            latent_sampling=dec_inputs[0],
            cond_true=c_inputs,
            y_true = x_inputs,
            y_pred = x_hat,
            dec_outputs = dec_outputs
        )
        if self.VAE_params.model_params.with_embedding:
            vae_args.update(dict(embedding_outputs=self.cond_embedding(c_inputs)))


        self.model.summary()
        self.blocks.append("model")

        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding.summary()
            self.blocks.append("cond_embedding")

        if self.VAE_params.model_params.with_Time2Vec:
            self.timetovec_model.summary()
            self.blocks.append("timetovec_model")

        self.encoder.summary()
        self.blocks.append("encoder")
        
        self.decoder.summary()
        self.blocks.append("decoder")

        # Training objectives settings
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)
        if len(list(self.VAE_params.training_params.loss.loss_weights.keys()))> 1:
            self.model.add_loss(self.VAE_params.training_params.loss._get_loss_function(**vae_args))

        print("Losses and associated weight involved in the model: ")
        [print(loss_key, " : ",
               self.VAE_params.training_params.loss.loss_weights[loss_key]) for loss_key in self.VAE_params.training_params.loss.losses.keys()]

        self.model.compile(loss=self.VAE_params.training_params.loss.losses["recon_loss"],
                           loss_weights=self.VAE_params.training_params.loss.loss_weights["recon_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)


class InteL_CVAE(AE_Model):
    def __init__(self, VAE_params):
        AE_Model.__init__(self, VAE_params = VAE_params)

    def build_model(self, VAE_params, custom_encoder_model=None, custom_decoder_model=None):
        """

        :param VAE_params: VAE_params class, subclasses of parameters needed to build each layers of the autoencoders and training paramaters to be used in the compile function
        :param custom_encoder_model: TF Model, to be used as encoder model in the graph
        :param custom_decoder_model: TF Model, to be used as decoder model in the graph
        :return: build graph and compile model in the CVAE Class
        """

        self.VAE_params.model_params = VAE_params.model_params
        if self.VAE_params.model_params.with_embedding :
            assert(len(self.VAE_params.model_params.cond_dims) + 1 == len(self.VAE_params.model_params.emb_dims))

        # getting the graph inputs
        x_inputs = Input(shape=(self.VAE_params.model_params.input_dims,), name="x_inputs")
        c_inputs = []
        context_inputs = []

        for i, c_dims in enumerate(self.VAE_params.model_params.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))
        
        if self.VAE_params.model_params.context_dims is not None:
            context_inputs = [Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")]
        inputs = [x_inputs] + c_inputs + context_inputs

        # Setting the AE architecture
        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding = EmbeddingBlock_model(input_dims=self.VAE_params.model_params.cond_dims,
                                                     emb_dims=self.VAE_params.model_params.emb_dims,
                                                     activation="relu", name="cond_emb", has_BN=False)

        if custom_encoder_model is not None:
            self.encoder = custom_encoder_model
        else:
            self.encoder = build_encoder_model(self, model_params=self.VAE_params.model_params)

        if custom_decoder_model is not None:
            self.decoder = custom_decoder_model
        else:
            self.decoder = build_decoder_model(self, model_params=self.VAE_params.model_params)

        # Model AE graph
        #encoding
        enc_outputs = self.encoder(inputs)

        if (self.VAE_params.model_params.nb_latent_components == 1):
            dec_inputs = [enc_outputs] + c_inputs
        else:
            y = Lambda(eval(self.VAE_params.model_params.reparametrize), name="reparametrizing_layer")(enc_outputs)

            z = Lambda(eval(self.VAE_params.model_params.intel_function), name="intermediate_layer")(y)

            dec_inputs = [z] + c_inputs

        #decoding
        dec_outputs = self.decoder(dec_inputs)

        if self.VAE_params.model_params.nb_decoder_outputs == 1:
            x_hat = dec_outputs
        else:
            x_hat = Lambda(eval(self.VAE_params.model_params.reparametrize), name="loglikelihood_layer")(dec_outputs)

        self.model = Model(inputs=inputs, outputs=x_hat, name="cvae")

        vae_args = dict(
            latent_components=enc_outputs,
            latent_sampling=dec_inputs[0],
            cond_true=c_inputs,
            y_true = x_inputs,
            y_pred = x_hat,
            dec_outputs = dec_outputs
        )
        if self.VAE_params.model_params.with_embedding:
            vae_args.update(dict(embedding_outputs=self.cond_embedding(c_inputs)))

        self.model.summary()
        self.blocks.append("model")

        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding.summary()
            self.blocks.append("cond_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

        # Training objectives settings
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)
        if len(list(self.VAE_params.training_params.loss.loss_weights.keys()))> 1:
            self.model.add_loss(self.VAE_params.training_params.loss._get_loss_function(**vae_args))
            self.model.add_metric(self.VAE_params.training_params.loss._get_loss_function(**vae_args), name='kl_loss', aggregation='mean')
        print("Losses and associated weight involved in the model: ")
        [print(loss_key, " : ",
               self.VAE_params.training_params.loss.loss_weights[loss_key]) for loss_key in self.VAE_params.training_params.loss.losses.keys()]

        self.model.compile(loss=self.VAE_params.training_params.loss.losses["recon_loss"],
                           loss_weights=self.VAE_params.training_params.loss.loss_weights["recon_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)
        
class GuidedCVAE(AE_Model):

    def __init__(self, VAE_params):
        AE_Model.__init__(self, VAE_params = VAE_params)

    def build_model(self, VAE_params, custom_encoder_model=None, custom_decoder_model=None):
        """

        :param VAE_params: VAE_params class, subclasses of parameters needed to build each layers of the autoencoders and training paramaters to be used in the compile function
        :param custom_encoder_model: TF Model, to be used as encoder model in the graph
        :param custom_decoder_model: TF Model, to be used as decoder model in the graph
        :return: build graph and compile model in the CVAE Class
        """
        self.VAE_params.model_params = VAE_params.model_params
        assert(len(self.VAE_params.model_params.encoder_dims) == len(self.VAE_params.model_params.input_dims))
        if self.VAE_params.model_params.with_embedding:
            assert(len(self.VAE_params.model_params.cond_dims) + 1 == len(self.VAE_params.model_params.emb_dims))

        # getting the graph inputs and setting the AE architecture
        cx_inputs = []
        c_inputs = []
        context_inputs = []
        c_modelparams = []
        c_encoders = []

        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding = EmbeddingBlock_model(input_dims=self.VAE_params.model_params.cond_dims,
                                                     emb_dims=self.VAE_params.model_params.emb_dims,
                                                     activation="relu", name="cond_emb", has_BN=False)

        for i, input_dim in enumerate(self.VAE_params.model_params.input_dims):
            if i == 0:
                x_inputs = Input(shape=(input_dim,), name=f"inputs_{i}")
                if custom_encoder_model is not None:
                    encoder_x = custom_encoder_model
                else:
                    x_modelparams = copy.deepcopy(VAE_params.model_params)
                    x_modelparams.input_dims = VAE_params.model_params.input_dims[i]
                    x_modelparams.encoder_dims = VAE_params.model_params.encoder_dims[i]
                    encoder_x = build_encoder_model(self, model_params=x_modelparams, name=f"_{i}")
            else:
                cx_inputs.append(Input(shape=(input_dim,), name=f"inputs_{i}"))
                c_modelparams.append(copy.deepcopy(VAE_params.model_params))
                c_modelparams[-1].input_dims = input_dim
                c_modelparams[-1].encoder_dims = self.VAE_params.model_params.encoder_dims[i]
                c_modelparams[-1].cond_dims = []
                c_modelparams[-1].encoder_type = "NNBlock_model"
                c_modelparams[-1].nb_latent_components = 1

                c_encoders.append(build_encoder_model(self, model_params=c_modelparams[-1], name=f"_{i}"))

        for i, c_dims in enumerate(self.VAE_params.model_params.cond_dims):
            c_inputs.append(Input(shape=(c_dims,), name="cond_inputs_{}".format(i)))

        if self.VAE_params.model_params.context_dims is not None:
            context_inputs = [Input(shape=(self.VAE_params.model_params.context_dims,), name="context_inputs")]
        inputs = [x_inputs] + cx_inputs + c_inputs + context_inputs

        self.encoder = build_guidedencoder_model(self, model_params=self.VAE_params.model_params,
                                                 x_encoder=encoder_x, list_condencoder=c_encoders)

        if custom_decoder_model is not None:
            self.decoder = custom_decoder_model
        else:
            self.decoder = build_decoder_model(self, model_params=x_modelparams)

        # Model AE graph
        #encoding
        enc_outputs = self.encoder(inputs)

        if (self.VAE_params.model_params.nb_latent_components == 1):
            dec_inputs = [enc_outputs] + c_inputs
        else:
            z = Lambda(eval(self.VAE_params.model_params.reparametrize), name="reparametrizing_layer")(enc_outputs)
            dec_inputs = [z] + c_inputs

        #decoding
        dec_outputs = self.decoder(dec_inputs)

        if self.VAE_params.model_params.nb_decoder_outputs == 1:
            x_hat = dec_outputs
        else:
            x_hat = Lambda(eval(self.VAE_params.model_params.reparametrize), name="loglikelihood_layer")(dec_outputs)

        self.model = Model(inputs=inputs, outputs=x_hat, name="cvae")

        vae_args = dict(
            latent_components=enc_outputs,
            latent_sampling=dec_inputs[0],
            cond_true=c_inputs,
            y_true=x_inputs,
            y_pred=x_hat,
            dec_outputs=dec_outputs
        )
        if self.VAE_params.model_params.with_embedding:
            vae_args.update(dict(embedding_outputs=self.cond_embedding(c_inputs)))


        self.model.summary()
        self.blocks.append("model")

        if self.VAE_params.model_params.with_embedding:
            self.cond_embedding.summary()
            self.blocks.append("cond_embedding")

        self.encoder.summary()
        self.blocks.append("encoder")

        self.decoder.summary()
        self.blocks.append("decoder")

        # Training objectives settings
        optimizer = self.VAE_params.training_params.optimizer(self.VAE_params.training_params.lr)
        if len(list(self.VAE_params.training_params.loss.loss_weights.keys())) > 1:
            self.model.add_loss(self.VAE_params.training_params.loss._get_loss_function(**vae_args))

        print("Losses and associated weight involved in the model: ")
        [print(loss_key, " : ",
               self.VAE_params.training_params.loss.loss_weights[loss_key]) for loss_key in self.VAE_params.training_params.loss.losses.keys()]

        self.model.compile(loss=self.VAE_params.training_params.loss.losses["recon_loss"],
                           loss_weights=self.VAE_params.training_params.loss.loss_weights["recon_loss"],
                           optimizer=optimizer, experimental_run_tf_function=False)













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
