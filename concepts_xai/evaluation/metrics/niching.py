import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from scipy.special import softmax
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def niche_completeness(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: trained decoder model to use for predicting the task
        labels from the concept data
    :return: Accuracy of predictor_model, evaluated on niches obtained from the
        provided concept and label data
    '''
    n_tasks = y_true.shape[1]
    # compute niche completeness for each task
    niche_completeness_list, y_pred_list = [], []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict_proba(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        y_pred_list.append(y_pred_niche[:, task])

    y_preds = np.vstack(y_pred_list).T
    y_preds = softmax(y_preds, axis=1)
    auc = roc_auc_score(y_true, y_preds, multi_class='ovo')

    result = {
        'auc_completeness': auc,
        'y_preds': y_preds,
    }
    return result


def niche_completeness_ratio(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche completeness ratio for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels
        from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated
        on niches and the accuracy of predictor_model evaluated on all concepts
    '''
    n_tasks = y_true.shape[1]

    y_pred_test = predictor_model.predict_proba(c_pred)
    if predictor_model.__class__.__name__ == 'Sequential':
        # get class labels from logits
        y_pred_test = y_pred_test > 0
    elif len(y_pred_test.shape) == 1:
        y_pred_test = y_pred_test[:, np.newaxis]

    # compute niche completeness for each task
    niche_completeness_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict_proba(niche)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche = y_pred_niche > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche = y_pred_niche[:, np.newaxis]

        # compute accuracies
        accuracy_base = accuracy_score(y_true[:, task], y_pred_test[:, task])
        accuracy_niche = accuracy_score(y_true[:, task], y_pred_niche[:, task])

        # compute the accuracy ratio of the niche w.r.t. the baseline
        # (full concept bottleneck) the higher the better (high predictive power
        # of the niche)
        niche_completeness = accuracy_niche / accuracy_base
        niche_completeness_list.append(niche_completeness)

    result = {
        'niche_completeness_ratio_mean': np.mean(niche_completeness_list),
        'niche_completeness_ratio': niche_completeness_list,
    }
    return result


def niche_impurity(c_pred, y_true, predictor_model, niches):
    '''
    Computes the niche impurity score for the downstream task
    :param c_pred: Concept data predictions, numpy array of shape
        (n_samples, n_concepts)
    :param y_true: Ground-truth task label data, numpy array of shape
        (n_samples, n_tasks)
    :param predictor_model: sklearn model to use for predicting the task labels
        from the concept data
    :return: Accuracy ratio between the accuracy of predictor_model evaluated on
        concepts outside niches and the accuracy of predictor_model evaluated on
        concepts inside niches
    '''
    n_tasks = y_true.shape[1]

    # compute niche completeness for each task
    y_pred_list = []
    for task in range(n_tasks):
        # find niche
        niche = np.zeros_like(c_pred)
        niche[:, niches[:, task] > 0] = c_pred[:, niches[:, task] > 0]

        # find concepts outside the niche
        niche_out = np.zeros_like(c_pred)
        niche_out[:, niches[:, task] <= 0] = c_pred[:, niches[:, task] <= 0]

        # compute task predictions
        y_pred_niche = predictor_model.predict_proba(niche)
        y_pred_niche_out = predictor_model.predict_proba(niche_out)
        if predictor_model.__class__.__name__ == 'Sequential':
            # get class labels from logits
            y_pred_niche_out = y_pred_niche_out > 0
        elif len(y_pred_niche.shape) == 1:
            y_pred_niche_out = y_pred_niche_out[:, np.newaxis]

        y_pred_list.append(y_pred_niche_out[:, task])

    y_preds = np.vstack(y_pred_list).T
    y_preds = softmax(y_preds, axis=1)
    auc = roc_auc_score(y_true, y_preds, multi_class='ovo')

    return {
        'auc_impurity': auc,
        'y_preds': y_preds,
    }


def niche_finding(c, y, mode='mi', threshold=0.5):
    n_concepts = c.shape[1]
    if mode == 'corr':
        corrm = np.corrcoef(np.hstack([c, y]).T)
        niching_matrix = corrm[:n_concepts, n_concepts:]
        niches = np.abs(niching_matrix) > threshold
    elif mode == 'mi':
        nm = []
        for yj in y.T:
            mi = mutual_info_classif(c, yj)
            nm.append(mi)
        nm = np.vstack(nm).T
        niching_matrix = nm / np.max(nm)
        niches = niching_matrix > threshold
    else:
        return None, None, None

    return niches, niching_matrix


def niche_impurity_score(
    c_soft,
    c_true,
    predictor_model_fn=None,
    predictor_train_kwags=None,
    delta_beta=0.05,
    test_size=0.2,
):
    """
    Returns the niche impurity score (NIS) of the given soft concept
    representations `c_soft` with respect to their corresponding ground truth
    concepts `c_true`. This value is higher if concepts encode unnecessary
    information from other concepts distributed across SUBSETS of soft concept
    representations, and lower otherwise.

    :param Or[np.ndarray, List[np.ndarray]] c_soft: Predicted set of "soft"
        concept representations by a concept encoder model applied to the
        testing data. This argument must be an np.ndarray with shape
        (n_samples, ..., n_concepts) where the concept representation may be
        of any rank as long as the last dimension is the dimension used to
        separate distinct concept representations. If concepts have distinct
        array shapes for their representations, then this argument is expected
        to be a list of `n_concepts` np.ndarrays where the i-th element in the
        list is an array with shape (n_samples, ...) containing the tensor
        representation of the i-th concept.
        Note that in either case we only require that the first dimension.
    :param np.ndarray c_true: Ground truth concept values in one-to-one
        correspondence with concepts in c_soft. Shape must be
        (n_samples, n_concepts).
    :param Function[(int,), sklearn-like Estimator] predictor_model_fn: A
        function generator that takes as an argument the number of
        the output target concept and produces an sklearn-like Estimator
        which one can train for predicting a concept given another concept's
        soft concept values. If not given then we will use a 3-layer ReLU MLP
        as our predictor.
    :param Dict[Any, Any] predictor_train_kwags: optional arguments to pass
        the estimator being when calling its `fit` method.
    :param float test_size: A value in [0, 1] indicating the fraction of the
        given data that will be used to evaluate the trained concept-based
        classifier. The rest of the data will be used for training said
        classifier.

    :returns float: A non-negative float in [0, 1] representing the degree to
        which individual concepts in the given representations encode
        unnecessary information regarding other concepts distributed across
        them.
    """
    (n_samples, n_concepts) = c_true.shape
    # finding niches for several values of beta
    niche_impurities = []

    if predictor_model_fn is None:
        predictor_model_fn = lambda n_concepts: MLPClassifier(
            (20, 20),
            random_state=1,
            max_iter=1000,
        )
    if predictor_train_kwags is None:
        predictor_train_kwags = {}

    # And estimate the area under the curve using the trapezoid method
    auc = 0
    prev_value = None
    classifier = predictor_model_fn(n_concepts=n_concepts)
    c_soft_train, c_soft_test, c_true_train, c_true_test = train_test_split(
        c_soft,
        c_true,
        test_size=test_size,
    )
    classifier.fit(c_soft_train, c_true_train, **predictor_train_kwags)

    for beta in np.arange(0.0, 1.0, delta_beta):
        niches, _ = niche_finding(
            c_soft_train,
            c_true_train,
            'corr',
            threshold=beta,
        )
        # compute impurity scores
        nis_score = niche_impurity(
            c_soft_test,
            c_true_test,
            classifier,
            niches,
        )
        niche_impurities.append(nis_score['auc_impurity'])
        # And update the area under the curve
        if prev_value is not None:
            auc += (prev_value + nis_score['auc_impurity']) * (delta_beta / 2)
        prev_value = nis_score['auc_impurity']

    return auc

