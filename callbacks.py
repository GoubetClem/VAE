import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from scipy import sparse
import numpy as np


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


def numpy_Gram_matrix(x, h):
    """Compute the normalized Gram matrix of a vector

    params:
    x -- array-like, vector on which compute the Gram matrix
    h -- float, window for kernel transformation

    """

    Dist_M = np.exp(-0.5 * np.square((x.reshape(-1, 1) - x.reshape(1, -1)) / h))  # / (np.sqrt(2*pi)*h)
    Norm = np.sqrt(np.diag(Dist_M).reshape(-1, 1) * np.diag(Dist_M).reshape(1, -1))
    return Dist_M / (Norm)


def numpy_make_Gram_matrix(A, h=4):
    d0 = A.shape[0]
    d1 = A.shape[1]
    A_tiled1 = np.tile(A.reshape(d0,1,d1), [1,d0,1])
    A_tiled2 = np.tile(A.reshape(1,d0,d1), [d0,1,1])
    gram_A = np.exp(-0.5 *np.sum(np.square(A_tiled1 - A_tiled2 ), axis=-1) / h)
    Norm = np.sqrt(np.diag(gram_A).reshape(-1, 1) * np.diag(gram_A).reshape(1, -1))

    return gram_A / Norm / A.shape[0]


def numpy_trace_normalize(A):
    trace_A = np.trace(A)
    return A / trace_A


def numpy_Renyi_entropy(A, alpha):
    """Compute the Renyi entropy analogy of a matrix based on its eigenvalues

    params:
    A -- array-like, matrix
    alpha -- float, Renyi entropy parameter
    """
    A[A * A.shape[0] < 1e-3] = 0
    A_sparse = sparse.csc_matrix(A)
    A_eigval, _ = np.abs(sparse.linalg.eigsh(A_sparse)) + 1e-6
    return np.log(np.sum((A_eigval) ** alpha)) / np.log(2) / (1 - alpha)


def nummpy_Shannon_entropy(A):
    A_eigval = np.abs(np.linalg.eigvalsh(A)) + 1e-6
    return -np.sum(A_eigval * np.log(A_eigval)) / np.log(2)


def Silverman_rule(h, n, d):
    return h * (n ** (-1 / (4 + d)))


class InformationHistory(Callback):
    """Instaure the callback to measure mutual information evolution between targeted layers during the training of an autoencoder

    :return: dict, mutual information between targeted layers of the VAE

    """

    def __init__(self, scale, alpha, dataset_train, cond_insert, infotoeval=["XZ", "XX'"], period=None, printlogger=False):

        """

        :param h:
        :param alpha:
        :param dataset_train:
        :param infotoeval: options are XX', XZ, X(C+Z), CZ, CEmb, ZEmb(if applicable)
        :param period:
        """
        self.alpha = alpha
        self.period = period
        self.dataset_train = dataset_train
        self.memory_epoch = 0
        self.infotoeval=infotoeval
        self.printlogger = printlogger
        self.cond_insert = cond_insert

        if isinstance(dataset_train, list):
            x_inputs = dataset_train[0]
        else:
            x_inputs = dataset_train

        self.sigma = scale
        self.gram_x = numpy_make_Gram_matrix(x_inputs, self.sigma)
        self.data_entropy = numpy_Renyi_entropy(numpy_trace_normalize(self.gram_x), alpha)

        if "C" in "".join(infotoeval):
            if isinstance(dataset_train, list) & len(dataset_train[1:]) >= 2 :
                self.gram_c = numpy_make_Gram_matrix(np.c_[dataset_train[1:]], self.sigma)
            else:
                self.gram_c = numpy_make_Gram_matrix(dataset_train[1], self.sigma)

            self.context_entropy = numpy_Renyi_entropy(numpy_trace_normalize(self.gram_c), alpha)

        self.MI={}
        for key in infotoeval:
            self.MI[key] = []


    def on_epoch_end(self, epoch, logs={}):

        self.memory_epoch += 1

        if self.period is None:
            do_callback = True
        else:
            if self.memory_epoch in self.period:
                do_callback = True
            else:
                do_callback = False

        if do_callback:
            if "Z" in "".join(self.infotoeval):
                lays_enc = self.model.get_layer('encoder')
                latent_mu = lays_enc.predict(self.dataset_train)[0]
                gram_z = numpy_make_Gram_matrix(latent_mu, self.sigma)

            if "X'" in "".join(self.infotoeval):
                x_hat = self.model.predict(self.dataset_train)
                gram_xhat = numpy_make_Gram_matrix(x_hat, self.sigma)

            if "Emb" in "".join(self.infotoeval):
                lays_emb = self.model.get_layer(self.cond_insert[0]).get_layer('cond_emb')
                emb = lays_emb.predict(self.dataset_train[1:])
                gram_emb = numpy_make_Gram_matrix(emb, self.sigma)

            if "XX'" in self.infotoeval:
                self.MI["XX'"].append(self.data_entropy + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_xhat), self.alpha) - numpy_Renyi_entropy(
                    numpy_trace_normalize(self.gram_x * gram_xhat), self.alpha))

            if "XZ" in self.infotoeval:
                self.MI["XZ"].append(self.data_entropy + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_z), self.alpha) - numpy_Renyi_entropy(
                    numpy_trace_normalize(self.gram_x * gram_z), self.alpha))

            if "CZ" in self.infotoeval:
                self.MI["CZ"].append(self.context_entropy + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_z), self.alpha) - numpy_Renyi_entropy(
                    numpy_trace_normalize(self.gram_c * gram_z), self.alpha))

            if "X(C+Z)" in self.infotoeval:
                self.MI["X(C+Z)"].append(self.data_entropy + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_z * self.gram_c), self.alpha) - numpy_Renyi_entropy(
                    numpy_trace_normalize(self.gram_x * self.gram_c * gram_z), self.alpha))

            if "CEmb" in self.infotoeval:
                self.MI["CEmb"].append(self.context_entropy + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_emb), self.alpha)- numpy_Renyi_entropy(
                    numpy_trace_normalize(self.gram_c * gram_emb), self.alpha))

            if "ZEmb" in self.infotoeval:
                self.MI["ZEmb"].append(numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_z), self.alpha) + numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_emb), self.alpha)- numpy_Renyi_entropy(
                    numpy_trace_normalize(gram_z * gram_emb), self.alpha))

            if self.printlogger:
                print("Mutual informations for epoch {} are ".format(
                    self.memory_epoch) + " ; ".join([x[0]+ " :  {:.3f}".format(x[1][-1]) for x in self.MI.items()]))
