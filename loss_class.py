from losses import *
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError

class VAE_Loss():

    def __init__(self, loss_weights={"recon_loss" : 1.}, recon_loss="mae", custom_loss=None):
        self.loss_weights=loss_weights
        self.custom_loss = custom_loss

        self.losses={}
        if recon_loss=="mae":
            self.losses["recon_loss"] = MeanAbsoluteError
        elif recon_loss=="mse":
            self.losses["recon_loss"] = MeanSquaredError

        if "kl_loss" in loss_weights.keys():
            self.losses["kl_loss"] = build_kl_loss

        if "info_loss" in loss_weights.keys():
            self.losses["info_loss"] = build_info_loss

        if "entropy_loss" in loss_weights.keys():
            self.losses["entropy_loss"] = build_entropy_loss

        if custom_loss is not None:
            for key in custom_loss.keys():
                self.losses[key] = custom_loss[key]["function"]

    def get_dict(self, **kwargs):
        """
        DEFINE THIS AS YOU WANT
        :param x:
        :param z:
        :param hat_x:
        :return:
        """
        dict_args_ = {}

        dict_args_["kl_loss"] = {"latent_components" : kwargs["latent_components"]}
        dict_args_["info_loss"] = {"latent_mu" : kwargs["latent_components"][0],
                                  "latent_sampling" : kwargs["latent_sampling"]}
        dict_args_["entropy_loss"] = {"cond_true" : kwargs["cond_true"], "latent_mu" : kwargs["latent_components"][0]}

        if self.custom_loss is not None:
            for key in self.custom_loss.keys():
                dict_args_[key] = {arg : self.custom_loss[key]["args"][arg] for arg in self.custom_loss[key]["args"].keys()}

        return dict_args_

    def _get_loss_function(self, **kwargs):
        """
        DO NOT TOUCH THIS
        :param x:
        :param z:
        :param hat_x:
        :return:
        """
        dict_ = self.get_dict(**kwargs)

        def vae_loss(y_true, y_pred):
            loss_call = []
            for key in self.losses.keys():
                this_kwargs = {}
                if key in dict_:
                    this_kwargs = dict_[key]
                loss_key = self.losses[key]
                loss_call.append(self.loss_weights[key] * loss_key()(y_true=y_true, y_pred=y_pred, **this_kwargs))

            return K.sum(K.concatenate(loss_call))

        return vae_loss

