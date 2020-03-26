from losses import *
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError


class VAELoss():
    """
    Class to build custom VAE loss in an aggregative way
    """

    def __init__(self, loss_weights={"recon_loss": 1.}, recon_loss="mae", custom_loss=None):
        """

        :param loss_weights: dict, {<str>:<float>} name and associated weight of the considered loss
        :param recon_loss: "mae" or "mse", whether to consider a L1 or L2 reconstruction loss
        :param custom_loss: dict, {"name_1":{"function" : <function>, "args":{}}} dictionary of custom losses externally built
        """

        self.loss_weights = loss_weights
        self.custom_loss = custom_loss
        self.prior_mu = K.variable(0.)
        self.log_prior_sigma = K.variable(0.)

        self.losses = {}
        if "recon_loss" in loss_weights.keys():
            if recon_loss == "mae":
                self.losses["recon_loss"] = MeanAbsoluteError()
            elif recon_loss == "mse":
                self.losses["recon_loss"] = MeanSquaredError()
            else:
                raise ValueError("Unknown reconstruction loss type. Try 'mae' or 'mse'")

        if "kl_loss" in loss_weights.keys():
            self.losses["kl_loss"] = build_kl_loss

        if "info_loss" in loss_weights.keys():
            self.losses["info_loss"] = build_mmd_loss

        if "entropy_loss" in loss_weights.keys():
            self.losses["entropy_loss"] = build_entropy_loss

        if custom_loss is not None:
            for key in custom_loss.keys():
                self.losses[key] = custom_loss[key]["function"]


    def get_dict(self, **kwargs):
        """
        CAN BE MODIFIED CONSIDERING NEW OUTPUTS OF THE MODEL
        :param kwargs: references to pointers of  outputs in the graph (see models.py)
        :return: dict, dictionary of kwargs to be used in each referenced loss
        """

        dict_args_ = {}

        dict_args_["kl_loss"] = {"latent_components": kwargs["latent_components"], "prior_mu" : self.prior_mu,
                                 "log_prior_sigma" : self.log_prior_sigma}
        dict_args_["info_loss"] = {"latent_mu": kwargs["latent_components"][0],
                                   "latent_sampling": kwargs["latent_sampling"]}
        dict_args_["entropy_loss"] = {"cond_true": kwargs["cond_true"], "latent_mu": kwargs["latent_components"][0]}

        if self.custom_loss is not None:
            for key in self.custom_loss.keys():
                dict_args_[key] = {arg: self.custom_loss[key]["args"][arg] for arg in self.custom_loss[key]["args"].keys()}

        return dict_args_


    def _get_loss_function(self, **kwargs):
        """
        DO NOT TOUCH THIS

        """
        dict_ = self.get_dict(**kwargs)

        def vae_loss(y_true, y_pred):
            loss_call = []
            for key in self.losses.keys():
                this_kwargs = {}
                if key in dict_:
                    this_kwargs = dict_[key]
                loss_key = self.losses[key]
                loss_call.append(self.loss_weights[key] * loss_key(y_true=y_true, y_pred=y_pred, **this_kwargs))

            return K.sum(K.stack(loss_call))

        return vae_loss
