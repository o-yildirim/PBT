import numpy as np

np.float = float

from sklearn.metrics import f1_score
from skmultilearn.dataset import load_from_arff


def get_dataset(file):
    datasets = {
        '20NG.arff': 20,
        'Bibtex.arff': 159,
        'Corel5k.arff': 374,
        'Corel16k.arff': 153,
        'Delicious.arff': 983,
        'Enron.arff': 53,
        'Mediamill.arff': 101,
        'Reuters.arff': 103,
        'Stackex-chemistry.arff': 175,
        'Stackex_cooking.arff': 400,
        'Stackex_cs.arff': 274,
        'Stackex_philosophy.arff': 233,
    }

    if file[-5:] == ".arff":
        n_labels = datasets[file]
        X, Y = load_from_arff(f'./datasets/{file}', label_count=n_labels)
        X = X.toarray()
        Y = Y.toarray()
        return n_labels, X, Y


def ex_based_acc(y_true_vec, y_pred_vec):
    numerator = 0
    denumerator = 0

    for i in range(len(y_true_vec)):
        numerator += (y_true_vec[i] & y_pred_vec[i])
        denumerator += (y_true_vec[i] | y_pred_vec[i])

    if numerator == 0 or denumerator == 0:
        return 0.0

    return numerator / denumerator


def ex_based_prec(y_true_vec, y_pred_vec):
    numerator = 0
    denumerator = 0
    for i in range(len(y_true_vec)):
        numerator += (y_true_vec[i] & y_pred_vec[i])
        denumerator += y_pred_vec[i]

    if numerator == 0 or denumerator == 0:
        return 0.0

    return numerator / denumerator


def ex_based_recall(y_true_vec, y_pred_vec):
    numerator = 0
    denumerator = 0
    for i in range(len(y_true_vec)):
        numerator += (y_true_vec[i] & y_pred_vec[i])
        denumerator += y_true_vec[i]

    if numerator == 0 or denumerator == 0:
        return 0.0

    return numerator / denumerator


def ex_based_f1(ex_recall, ex_prec):
    if ex_prec == 0 and ex_recall == 0:
        return 0.0

    return (2 * ex_prec * ex_recall) / (ex_prec + ex_recall)


def hamming_score(y_true_vec, y_pred_vec):
    numerator = 0
    for i in range(len(y_true_vec)):
        if y_true_vec[i] == y_pred_vec[i]:
            numerator += 1
    return numerator


def compute_metrics_dataset_online(y_true, y_pred):
    n = len(y_true)
    acc = 0
    rec = 0
    prec = 0
    hamming_sc = 0

    for i in range(n):
        y_true_list = y_true[i]
        y_pred_list = y_pred[i].tolist()

        acc += ex_based_acc(y_true_vec=y_true_list, y_pred_vec=y_pred_list)
        rec += ex_based_recall(y_true_vec=y_true_list, y_pred_vec=y_pred_list)
        prec += ex_based_prec(y_true_vec=y_true_list, y_pred_vec=y_pred_list)
        hamming_sc += hamming_score(y_true_vec=y_true_list, y_pred_vec=y_pred_list)
    acc /= n

    rec /= n
    prec /= n
    f1 = ex_based_f1(rec, prec)

    hamming_sc /= (n * len(y_pred[0]))

    microF1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    return {'accuracy': acc, 'recall': rec, 'precision': prec, 'hamming_score': hamming_sc, 'f1_score': f1,
            'micro-f1': microF1}
