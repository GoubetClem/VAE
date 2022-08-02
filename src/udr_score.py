"""
 Implementation of the UDR score developped in the paper: "Unsupervised
Model Selection for Variational Disentangled Representation Learning"
(https://arxiv.org/abs/1905.12614)

This implementation is inspired by the disentanglement_lib of Olivier Bachem.
 """

from attr import s
import numpy as np
from scipy import rand
from sklearn.linear_model import Lasso
from sklearn import preprocessing
import sys

from metrics import evaluate_latent_code

def corr_mat_lasso(x1, x2): 
    """Computes correlation matrix of two representations using Lasso Regression.

    Args:
        x1: (batch_size, z_dim)-shape numpy array 
        x2: (batch_size, z_dim)-shape numpy array 
        random_state: int used to control the randomness during training

    Returns:
        corr_mat: np.(z_dim, z_dim)-shape numpy array with the correlation coefficients obtained with a lasso regression between x1 and x2.
            Elements of x1 correspond to axis 0 and elements of x2 correspond to axis 1.
    """
    assert x1.shape == x2.shape, 'Please give two arrays of the same shape as input'
    lasso = Lasso(alpha=0.1).fit(x1, x2)
    corr_mat = np.transpose(np.absolute(lasso.coef_))
    return corr_mat

def compute_multi_gaussian_kl(data_encoded):
    """Computes the mean Kullback-Leibler divergence in each latent dimension.

    Args: 
        data_encoded: list that contains encoded dataset for each model we want to compare.

    Return:
        list: list that contains the mean KL divergence in each latent dimension and for each model.  
    """
    n_seed_models = data_encoded.shape[1]
    n_hyper_models = data_encoded.shape[0]
    return np.array([[np.mean(0.5 * (np.square(data_encoded[i,j][0]) + np.exp(data_encoded[i,j][1]) - data_encoded[i,j][1] - 1),axis=0) for j in range(n_seed_models)] for i in range(n_hyper_models)])

def multi_model_encoding(data_encoded):
    return data_encoded[:,:,0], compute_multi_gaussian_kl(data_encoded)

    
def relative_strength_disentanglement(corr_mat):
  """Computes disentanglement using relative strength score."""
  r_a = np.power(np.ndarray.max(corr_mat, axis=0), 2)
  r_b = np.power(np.ndarray.max(corr_mat, axis=1), 2)
  
  score_x = np.nanmean(np.nan_to_num(r_a / np.sum(corr_mat, axis=0), 0))
  score_y = np.nanmean(np.nan_to_num(r_b / np.sum(corr_mat, axis=1), 0))
  return (score_x + score_y) / 2

def compute_udr_score(data_encoded,
                      P,
                      verbose = 0,
                      correlation_matrix="lasso",
                      filter_low_kl=True,
                      include_raw_correlations=True,
                      kl_filter_threshold=0.01):

    """Computes the UDR score as defined in the paper for each model given in input

    Args: 
        data_encoded: array that contains encoded dataset for each model trained where 
            the models in axis 0 are trained with the same seed and the models in axis 1
            have the same hyperparameters.
        P: int that represents the number of other trained models with the same 
            hyperparameters but different seeds that we sample in the 2nd step.
        correlation_matrix: method used to compute correlation matrix.
        filter_low_kl: boolean that precizes if we want to filter low informative KL.
        kl_filter_threshold: threshold at which a KL is or is not informative. 
        include_raw_correlations: bool that returns the raw correlations if true. 
    
    Returns: 
        score_dict: dict that contains the raw correlations between models if 
            include_raw_correlations=True, the pairwise disentanglement scores
            and the final scores for each model.
    """
    
    # They chose in the paper to sample P <= S others models with the same hyperparameters
    # but different seeds. /!\ In fact, P <= S-1 because we don't want to compute the 
    # similarities between the latent representations of the same model. 
    import random 

    models_latent_representation, kl_div = multi_model_encoding(data_encoded)
    n_seed_models = models_latent_representation.shape[1]
    n_hyper_models = models_latent_representation.shape[0]
    latent_dim = models_latent_representation[0][0].shape[1]

    if P > (n_seed_models- 1): 
        raise ValueError('We can sample only P <= (the number of models with the same hyperparameters but different seeds) -1') 

    # Recall: the array model_latent_representation[i,j] contains the encoded data
    #         with the model with the i-th hyperparameters set and the j-th seed. 
    global_corr_mat = np.zeros((n_hyper_models, n_seed_models, P, latent_dim, latent_dim))

    # Normalize and remove uninformative latents with the computation 
    # of the kl divergence 
    kl_mask = np.ones((n_hyper_models, n_seed_models, latent_dim), dtype=bool)
    disentanglement = np.zeros((n_hyper_models, n_seed_models, P, 1))
    for i in range(n_hyper_models):
        for j in range(n_seed_models): 
    
            scaler = preprocessing.StandardScaler()
            scaler.fit(models_latent_representation[i,j])
            models_latent_representation[i,j] = scaler.transform(models_latent_representation[i,j])

            models_latent_representation[i,j] = models_latent_representation[i,j]

            kl_mask[i,j] = (kl_div[i,j] > kl_filter_threshold)
    
            # Compute of the UDR(i,j) score for each model i,j and the model i,p where j!=p
            idx_seed_models = np.arange(n_seed_models).tolist()
            idx_seed_models.remove(j)
            sample_seed_models = random.sample(idx_seed_models, P)
            for idp, p in enumerate(sample_seed_models):
                if verbose:
                    sys.stdout.write('\r')
                    verbose_variable = (n_seed_models * P * i + P * j + idp + 1)/(P * n_hyper_models * n_seed_models)
                    sys.stdout.write("[%-50s] %d%%" % ('='*int(50*verbose_variable), 100*verbose_variable))
                    sys.stdout.flush()
                # The paper argues that results are slightly better while using lasso correlation matrix.
                # TODO: Look if there are other ways to compute relevant correlation matrix. 
                if correlation_matrix == "lasso":
                    corr_mat = corr_mat_lasso(models_latent_representation[i,j], models_latent_representation[i,p])
                #else:
                    # corr_mat = spearman_correlation_conv(models_latent_representation[i],
                    #                                      models_latent_representation[j])

                global_corr_mat[i, j, idp, :, :] = corr_mat
                if filter_low_kl:
                    corr_mat = corr_mat[kl_mask[i,j], ...][..., kl_mask[i,p]]
                disentanglement[i, j, idp] = relative_strength_disentanglement(corr_mat)

    scores_dict = {}
    if include_raw_correlations:
        scores_dict["raw_correlations"] = global_corr_mat.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    # Computation of the UDR score for each model
    model_scores = np.zeros((n_hyper_models, n_seed_models))
    for i in range(n_hyper_models):
        for j in range(n_seed_models):
            model_scores[i,j] = np.median(disentanglement[i,j,:])

    scores_dict["model_scores"] = model_scores

    return scores_dict

def display_udr_scores(data_encoded, 
                       udr_scores, 
                       models_name, 
                       factorMatrix, 
                       factorDesc, 
                       hyperparam,
                       palette='autumn_r',
                       udr=False):

    import pandas as pd
    import matplotlib.pyplot as plt 
    import seaborn as sns


    new = {}
    scores = {}
    tmp=[]
    progressbar_width = 2 * len(models_name[0])*len(models_name)
    for idx, model in enumerate(models_name):
        for i in range(len(model)):
            new[model[i]]=[]
            new[model[i]].append(idx)
            new[model[i]].append(int(model[i].split('_')[-1][4:]))
            strbarwidth = '{}{}{}\r'.format(
                (2*(i + idx * len(model)+1) * '\u001b[47m  '),
                ((progressbar_width - 2*(i + idx * len(model)+1)) * '\u001b[40m  '),
                (('\u001b[0m - {:0.2f}'.format(((2*(i + idx * len(model)+1)) * (100/progressbar_width))) + '% - Model n°{}/{}'.format(i + idx*len(model)+1, len(model)*len(models_name))))
             )
            print(strbarwidth ,end = '')
            x_encoded = data_encoded[idx, i, 0]
            scores[model[i]], _ = evaluate_latent_code(x_reduced=x_encoded, 
                                                       factorMatrix=factorMatrix,
                                                       factorDesc=factorDesc)
            if udr: 
                tmp.append([udr_scores[idx, i], 'UDR', new[model[i]][0],new[model[i]][1]])
            tmp.append([scores[model[i]]['mean_disentanglement'], 'DCI Disent', new[model[i]][0], new[model[i]][1]])
            scores[model[i]].pop('disentanglement')
            scores[model[i]].pop('mean_disentanglement')
            
            for met in scores[model[i]]:
                tmp.append([np.mean(scores[model[i]][met]), met.upper(), new[model[i]][0], new[model[i]][1]])
            
        
    
    df = pd.DataFrame(tmp, columns=['score','metric','hyperparam','seed'])

    # Converting dtype 
    df['score'] = df['score'].astype('float32')
    df['hyperparam'] = df['hyperparam'].astype(int)
    df['seed'] = df['seed'].astype(int)

    """hyperparam = pd.DataFrame(data=hyperparam)
    print("\nPlease find the summary table of hyperparameters below:")
    print(hyperparam.to_string(),'\n')"""
    
    fig, ax = plt.subplots(figsize=(15,10))
    sns.boxplot(x='metric',y='score',hue='hyperparam', data=df, palette=palette, ax=ax)
    ax.set_ylim(0,1)
    plt.show()

    return df









""" OLD VERSION 
def sample_observations_batch(data, batch_size, guided=False):
    
    Sample a unique mini-batch of observations from the training dataset. 

    Args: 
        data: the dataset that has been used for the model training.
        batch_size: size of batches of data
    
    Return:
        batch: dict that contains a well-formated sample of data that can be used for encoding
    

    n = data[0].shape[0]
    assert n >= batch_size, 'Please set a batch_size smaller than number of samples in the dataset'
    rand_indexes = np.arange(n)
    np.random.shuffle(rand_indexes)
    df = data.copy()
    for i in range(len(data)):
        df[i] = data[i][rand_indexes]
        if guided: 
            name_inputs = ['inputs_'+str(i) for i in range(len(data))]
        else:
            name_inputs = ['enc_inputs']+['enc_cond_inputs_'+str(i) for i in range(len(data)-1)]
    t = {name_inputs[j]:np.array(df[j][:batch_size]) for j in range(len(data))}
    return t
 

def sample_latent_batch(data, list_encoders, batch_size, guided=False):
    
    Sample a unique mini-batch of latent representation from the training dataset. 

    Args: 
        data: the dataset that has been used for the model training.
        batch_size: size of batches of data
    
    Return:
        list: list that contains a batch of [z_mean, z_log_var] for each model
    
    obs = sample_observations_batch(data, batch_size,guided)
    return [encoder(obs)for encoder in list_encoders]



def multi_model_encoding(data, list_encoders, batch_size, guided=False, verbose=1):
    
    Encode the entire dataset for each encoder in list_encoders. 

    Args:
        data: the dataset that has been used for the model training.
        list_encoders: a list with the encoder of each model we want to compare. 
        batch_size: size of batches of latent vectors.  
    
    Returns: 
        data_encoded: dict with n_models=len(list_encoders) key and the latent representation for each model j 
            data_encoded[j] = (N_sample, z_dim_j)-shape array 
        kl_div: dict with the average Kullback-Leibler divergence for each latent dimension 
    
    
    # Initialization
    n = data[0].shape[0]
    data_encoded 
    batches_size = [batch_size]*int(n/batch_size)+[n%batch_size]
    n_batch = len(batches_size)
    n_encoders = len(list_encoders)
    data_encoded = {j: np.zeros((n,4)) for j in range(len(list_encoders))}            
    kl_div = {j: np.zeros((n_batch,4)) for j in range(len(list_encoders))} 
    batch_size_next = 0
    


    # The model from Google Deep Mind suppose that batch_size is a divisor of the number of training data. 
    # We raised this restriction to be able to work with a prime number of training data. 
    for i, bs in enumerate(batches_size):
        latent_batch = sample_latent_batch(data, list_encoders, bs, guided)
        # latent_batch[j] is the latent representation obtained with the j-th encoder.

        for j in range(n_encoders):
                
                #z_dim_j = latent_batch[j][0].shape[1]
                #data_encoded[j] = np.zeros((n, z_dim_j))
                #kl_div[j] = np.zeros((n_batch, z_dim_j))
            if verbose:
                sys.stdout.write('\r')
                verbose_variable = (n_encoders * i + j + 1)/(n_batch * n_encoders)
                sys.stdout.write("[%-50s] %d%%" % ('='*int(50*verbose_variable), 100*verbose_variable))
                sys.stdout.flush()

            kl_div[j][i,:] = compute_gaussian_kl(latent_batch[j][0],latent_batch[j][1])
            data_encoded[j][i*batch_size_next:i*batch_size_next + bs,:] = latent_batch[j][0]
        batch_size_next = bs  
    print(kl_div)

    return data_encoded, [np.mean(kl_div[j], axis=0) for j in kl_div.keys()]

def compute_udr_sklearn(data_encoded,
                        correlation_matrix="lasso",
                        filter_low_kl=True,
                        include_raw_correlations=True,
                        kl_filter_threshold=0.01):

    
    models_latent_representation, kl_div = multi_model_encoding(data_encoded)

    num_models = len(models_latent_representation)

    latent_dim = models_latent_representation[0].shape[1]
    global_corr_mat = np.zeros((num_models, num_models, latent_dim, latent_dim))

    # Normalize and calculate mask based off of kl divergence to remove
    # uninformative latents.
    kl_mask = []
    for i in range(num_models):
        scaler = preprocessing.StandardScaler()
        scaler.fit(models_latent_representation[i])
        models_latent_representation[i] = scaler.transform(models_latent_representation[i])
        models_latent_representation[i] = models_latent_representation[i] * np.greater(kl_div[i], 0.01)
        kl_mask.append(kl_div[i] > kl_filter_threshold)

    # Compute of the UDR(i,j) score for each model i and j where i!=j
    disentanglement = np.zeros((num_models, num_models, 1))
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue
            
            # The paper argues that results are slightly better while using lasso correlation matrix.
            # TODO: Look if there are other ways to compute relevant correlation matrix. 
            if correlation_matrix == "lasso":
                corr_mat = corr_mat_lasso(models_latent_representation[i], models_latent_representation[j])
            #else:
                #corr_mat = spearman_correlation_conv(models_latent_representation[i],
                                                        #models_latent_representation[j])

            global_corr_mat[i, j, :, :] = corr_mat
            if filter_low_kl:
                corr_mat = corr_mat[kl_mask[i], ...][..., kl_mask[j]]
            disentanglement[i, j] = relative_strength_disentanglement(corr_mat)

    scores_dict = {}
    if include_raw_correlations:
        scores_dict["raw_correlations"] = global_corr_mat.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    # Computation of the UDR score for each model
    model_scores = []
    for i in range(num_models):
        # We delete the i-th score because we do not want the UDR_{i,i} score in the median.
        # The UDR(i,i) are still equal to 0 for all i.
        disentanglement_evaluation = np.delete(disentanglement[:, i], i)
        model_scores.append(np.median(disentanglement_evaluation)) 

    scores_dict["model_scores"] = model_scores

    return scores_dict

"""