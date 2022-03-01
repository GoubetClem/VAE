from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt


def disentanglement_quantification(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3,
                                   normalize_information=False):
    """criteria based on "A Framework for the Quantitative Evaluation of Disentangled Representations", Eastwood and Williams (2018)

    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of conditions names and types
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    normalize_information -- Boolean, whether to normalize informativeness results with the minimum obtained with a random projection

    :return: final_evaluation -- dict, dict of metrics values
             importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    assert algorithm == 'RandomForest' or algorithm == 'GradientBoosting'
    if algorithm == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier as clf
        from sklearn.ensemble import RandomForestRegressor as reg
    else:
        from sklearn.ensemble import GradientBoostingClassifier as clf
        from sklearn.ensemble import GradientBoostingRegressor as reg

    z_dim = x_reduced.shape[1]
    n_factors = factorMatrix.shape[1]
    evaluation = {}
    evaluation['informativeness'] = {}
    evaluation['importance_variable'] = {}
    final_evaluation = {}
    # estimation of importance of the latent code variables for each factor using random forest attribut of feature importances
    for i, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            estimator = clf(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:, i], cv=cv, return_estimator=True,
                                        scoring='f1_macro')
        else:
            estimator = reg(n_estimators=100)
            cv_results = cross_validate(estimator, x_reduced, factorMatrix[:, i], cv=cv, return_estimator=True,
                                        scoring='r2')

        if normalize_information:
            x_reduced_random = np.random.rand(x_reduced.shape[0], 1)
            if (factor_type == 'category'):
                estimator_random = clf(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:, i], cv=cv,
                                                   return_estimator=True, scoring='f1_macro')
            else:
                estimator_random = reg(n_estimators=100)
                cv_results_random = cross_validate(estimator_random, x_reduced_random, factorMatrix[:, i], cv=cv,
                                                   return_estimator=True, scoring='r2')

        if normalize_information:
            min_info = np.mean(cv_results_random['test_score'])
            results_info = np.nan_to_num((np.mean(cv_results['test_score']) - min_info) / (1 - min_info))
        else:
            results_info = np.mean(cv_results['test_score'])

        evaluation['informativeness'][name] = max(results_info, 0)
        importance_P = np.concatenate([esti.feature_importances_.reshape(-1, 1) for esti in cv_results['estimator']],
                                      axis=1)
        evaluation['importance_variable'][name] = np.mean(importance_P, axis=1)

    final_evaluation['informativeness'] = np.asarray(
        [evaluation['informativeness'][name] for name in factorDesc.keys()])

    importance_matrix = np.concatenate(
        [evaluation['importance_variable'][name].reshape(-1, 1) for name in factorDesc.keys()], axis=1)
    importance_matrix_norm = np.apply_along_axis(lambda x: x / np.sum(x), 1, importance_matrix)
    
    disentangled_measures = 1 + np.sum(
        importance_matrix_norm * np.log(importance_matrix_norm + 1e-10),
        axis=1) / np.log(n_factors)
    
    compactness_measures = 1 + np.sum(importance_matrix * np.log(importance_matrix + 1e-10), axis=0) / np.log(z_dim)

    weights_predictonefactor = np.sum(importance_matrix, axis=1) / np.sum(importance_matrix)
    weighted_disentanglement = np.sum(disentangled_measures * weights_predictonefactor)

    final_evaluation['disentanglement'] = disentangled_measures.ravel()
    final_evaluation['compactness'] = compactness_measures.ravel()
    final_evaluation['mean_disentanglement'] = weighted_disentanglement.ravel()

    return final_evaluation, importance_matrix


def compute_mig(x_reduced, factorMatrix, factorDesc, batch=None):
    """criterion Mutual Information Gap implementation based on "Isolating Sources of Disentanglement in Variational Autoencoders", Chen (2018);
       inspiration from disentanglement_lib of Olivier Bachem.

    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of genertaive/explicative names and types
    factorMatrix -- array-like, array explicative factors values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation

    :return: mig -- float, MIG average value across the factors
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index, :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    entropy = np.zeros(ys.shape[1])
    for j, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            m[:, j] = mutual_info_classif(latent, ys[:, j]).T
            entropy[j] = mutual_info_classif(ys[:, j].reshape(-1, 1), ys[:, j]).ravel()
        else:
            m[:, j] = mutual_info_regression(latent, ys[:, j]).T
            entropy[j] = mutual_info_regression(ys[:, j].reshape(-1, 1), ys[:, j]).ravel()

    sorted_m = np.sort(m, axis=0)
    mig = np.divide(sorted_m[-1, :] - sorted_m[-2, :], entropy)

    return mig


def compute_modularity(x_reduced, factorMatrix, factorDesc, batch=None):
    """criterion Modularity based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss", Ridgeway and Mozer (2018);
        inspiration from disentanglement_lib of Olivier Bachem.

    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of generative/explicative factors names and types
    factorMatrix -- array-like, array containing explicative factors values for the representation (columns in the keys order of factorDesc)
    batch -- whether to compute the MIG on a sliced part of the latent representation

    :return: modularity -- float, modularity score for the representation
    """

    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import mutual_info_regression

    if batch is None:
        train_size = x_reduced.shape[0]
    else:
        train_size = batch
    sample_index = np.random.choice(range(train_size), size=train_size, replace=False)
    latent = x_reduced[sample_index, :]
    ys = factorMatrix[sample_index, :]

    m = np.zeros((x_reduced.shape[1], factorMatrix.shape[1]))
    for j, name in enumerate(factorDesc.keys()):
        factor_type = factorDesc[name]
        if (factor_type == 'category'):
            m[:, j] = mutual_info_classif(latent, ys[:, j]).T
        else:
            m[:, j] = mutual_info_regression(latent, ys[:, j]).T

    sorted_m = np.r_[[np.eye(1, m.shape[1], k).ravel() for k in np.argmax(m, axis=1)]]
    t_i = m * sorted_m

    d_i = np.sum(np.square(m - t_i), axis=1) / np.square(np.max(m, axis=1)) / (factorMatrix.shape[1] - 1)

    return 1 - d_i


def evaluate_latent_code(x_reduced, factorMatrix, factorDesc, algorithm='RandomForest', cv=3, orthogonalize=True,
                       normalize_information=False):
    """ function to return a dict of implemented metrics which are informativeness, compactness, disentanglement, MIG and modularity

    params:
    x_reduced -- array-like, array containing the coordinates of the representation
    factorDesc -- dict, dict of generative/causal factors names and types
    factorMatrix -- array-like, array containing conditions values for the representation (columns in the keys order of factorDesc)
    algorithm -- the kind of estimator to make predictions with
    cv -- int, cross-validation generator or an iterable. Determines the cross-validation splitting strategy.
    orthogonalize -- Boolean, whether to fix the explicative axes of the representation on the coordinates dimensions
    normalize_information -- Boolean, whether to normalize informtiveness results with the minimum obtained with a random projection

    :return: final_evaluation -- dict, dict of metrics values
             importance_matrix -- array-like, importance matrix for latent dimensions (rows) to predict factors (columns)
    """
    if orthogonalize:
        from sklearn.decomposition import PCA
        ortho_proj = PCA(x_reduced.shape[1])
        x = ortho_proj.fit_transform(x_reduced)
    else:
        x = x_reduced

    final_evaluation, importance_matrix = disentanglement_quantification(x, factorMatrix, factorDesc,
                                                                         algorithm=algorithm, cv=cv,
                                                                         normalize_information=normalize_information)
    final_evaluation['mig'] = compute_mig(x, factorMatrix, factorDesc)
    final_evaluation['modularity'] = compute_modularity(x, factorMatrix, factorDesc)

    return final_evaluation, importance_matrix


def display_evaluation_latent_code(final_evaluation, z_dim, factorDesc):
    """A proposition of barplots to display results of the computed disentanglement metrics

    Args:
        final_evaluation (dict): dict of computed metrics values
        z_dim (int): number of latent dimensions
        factorDesc (dict): dict of generative/explicative factors names and types
    """
    
    if 'reconstruction_error' in final_evaluation.keys():
        for k,v in final_evaluation['reconstruction_error'].item():
            print(k, ' : ', v)

    fig = plt.figure(dpi=100,figsize=(10,8))

    plt.subplot(2, 3, 1)
    fig.subplots_adjust(hspace=0.5)
    plt.bar(factorDesc.keys(),final_evaluation['informativeness'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.ylim(top=1)
    for index,data in enumerate(final_evaluation['informativeness']):
        plt.text(x=index - 0.5, y =data+0.01 , s="%.2f"%data , fontdict=dict(fontsize=10))
    plt.title('Informativeness score : %.2f'%np.mean(final_evaluation['informativeness']))

    plt.subplot(2, 3, 2)
    plt.bar(np.arange(z_dim)+1, final_evaluation['disentanglement'])
    plt.xlabel('latent variables')
    plt.title('Disentanglement score : %.2f'%final_evaluation['mean_disentanglement']);

    plt.subplot(2, 3, 3)
    plt.bar(factorDesc.keys(),final_evaluation['compactness'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.title('Compactness')
    plt.tight_layout();
    
    plt.subplot(2, 3, 5)
    plt.bar(np.arange(z_dim)+1,1-final_evaluation['modularity'])
    plt.xlabel('latent variables')
    plt.title('Modularity score : %.2f'%np.mean(1-final_evaluation['modularity']));
    
    plt.subplot(2, 3, 6)
    plt.bar(factorDesc.keys(),final_evaluation['mig'])
    plt.xlabel('factors')
    plt.xticks(rotation=75)
    plt.title('Mutual Information Gap (MIG) : %.2f'%np.mean(final_evaluation['mig']))
    plt.tight_layout();
    
    plt.show()