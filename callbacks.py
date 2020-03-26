import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class callbackWeightLoss(Callback):  # to adapt the weights of the loss components
    # customize your behavior
    def __init__(self, lossattr="recon_loss", rate=0.002, minimum=0.001, logstart=500):
        self.lossattr = lossattr
        self.rate = rate
        self.minimum = minimum
        self.logstart = logstart

    def on_epoch_end(self, epoch, logs={}):
        if (epoch > self.logstart):

            weightVar = self.model.loss.loss_weights[self.lossattr]
            weight = K.get_value(weightVar)
            new_Weight = weight - self.rate * weight  # 0.99*np.cos(epoch/360*2*Pi)
            # if(new_Weight>=10000*self.beta ):
            #    new_Weight=100*self.beta
            if (new_Weight <= self.minimum):
                new_Weight = self.minimum
            K.set_value(weightVar, new_Weight)