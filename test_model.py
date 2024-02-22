import time

import numpy as np

np.float = float
from river import stream
from tqdm import tqdm

import Utilities
from PrioritizedBinaryTransformation import PrioritizedBinaryTransformation

if __name__ == '__main__':
    dataset = 'Corel5k.arff'
    n_labels, X, Y = Utilities.get_dataset(dataset)

    model = PrioritizedBinaryTransformation(n_labels=n_labels, t=1000)

    # Initialize data stream.
    ds = stream.iter_array(X, Y)

    # Initilaize arrays for the evaluation at the end.
    y_true_arr = []
    y_pred_arr = []

    pbar = tqdm(total=len(X))  # To keep track of the progress (progress bar).

    start_time = time.time()
    for x, y in ds:
        # Test for the current sample
        y_pred = model.predict_one(x)

        # Train with the current sample
        model.learn_one(x, y)

        y_true_arr.append(list(y.values()))  # Ground truth array (for evaluation at the end).
        y_pred_arr.append(y_pred.astype(int))  # Prediction array (for evaluation at the end).

        pbar.update(1)

    print(str(dataset) + " processed in " + str((time.time() - start_time)) + " seconds.")
    pbar.close()

    results = Utilities.compute_metrics_dataset_online(y_true_arr, y_pred_arr)
    print("Acc: " + "{:.3f} ".format(results["accuracy"]) + ", " + "Hamming Score: " + "{:.3f} ".format(
        results["hamming_score"]))
    print("Micro-F1: " + "{:.3f} ".format(results["micro-f1"]) + ", " + "Example-based F1: " + "{:.3f} ".format(
        results["f1_score"]))
