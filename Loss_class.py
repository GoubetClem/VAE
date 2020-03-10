from losses import *
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError

class VAE_Loss():

    def __init__(self, loss_weights={"recon_loss" : 1.}, recon_loss="mae", custom_loss=None):
        self.loss_weights=loss_weights

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
                self.losses[key] = custom_loss[key]


    def __call__(self, **kwargs):

        def vae_loss(y_true, y_pred):
            loss_call = []

            for key in self.losses.keys():
                loss_call.append(self.loss_weights[key] * self.losses[key](y_true, y_pred, **kwargs))

            return K.sum(K.concatenate(loss_call))

        return vae_loss

