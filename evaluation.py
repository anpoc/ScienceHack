from typing import List, Dict, Tuple, Callable

import numpy as np
from dataset import InvoiceBatchDataset
from tqdm import tqdm


def evaluate(predict: Callable, split: str = "train", n: int = 100) -> float:
    ds = InvoiceBatchDataset("data", split=split, min_n=5, max_n=15, size=n)
    print(len(ds))
    exact_matches = []
    accuracies = []
    chunk_scores = []
    for batch in tqdm(ds):
        pdf_path, y_true = batch
        y_pred = predict(pdf_path)
        scores = evaluate_predictions(y_true, y_pred)
        exact_matches.append(scores["exact_match"])
        chunk_scores.append(scores["chunk_score"])
        accuracies.append(scores["accuracy"])
    exact_match = sum(exact_matches) / len(exact_matches)
    chunk_score = sum(chunk_scores) / len(chunk_scores)
    accuracy = sum(accuracies) / len(accuracies)
    print(f"{split} results:")
    print(f"exact matches: {exact_match:.2f}")
    print(f"accuracy: {np.mean(accuracy):.2f}")
    print(f"chunk score: {chunk_score:.2f}")
    return exact_match, accuracy, chunk_score

def _chunks(arr: List[int]) -> List[Tuple[int, int]]:
    spans = []
    n = len(arr)
    for i, val in enumerate(arr):
        if val == -1:
            # if it is unknown, dont consider it a chunk
            continue
        if val == 1:
            j = i + 1
            while j < n and arr[j] == 0:
                j += 1
            spans.append((i, j))
    return spans

def evaluate_predictions(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, float]:
    """
    Compute standard token-level metrics plus chunk-level metrics.
    
    Returns
    -------
    dict with keys:
      accuracy, precision, recall, f1
      exact_match                – 1.0 if lists are identical, else 0.0
      chunk_precision, chunk_recall, chunk_f1
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Ground-truth and prediction lists must have equal length.")
    if not all(v in (0, 1) for v in y_true):
        raise ValueError("Ground-truth list must contain only 0 or 1.")
    if not all(v in (0, 1, -1) for v in y_pred):
        raise ValueError("Prediction list must contain only 0, 1, or -1.")

    exact_match = float(y_true == y_pred)
    true_chunks  = set(_chunks(y_true))
    pred_chunks  = set(_chunks(y_pred))
    correct_chunks = true_chunks & pred_chunks
    chunk_match = len(correct_chunks) / (1/2 * (len(pred_chunks) + len(true_chunks))) if pred_chunks else 0.0

    return {
        "exact_match": exact_match,
        "chunk_score": chunk_match,
        "accuracy": np.mean(np.array(y_true) == np.array(y_pred)),
    }

def evaluate_during_training(predict: Callable, split: str = "train", n: int = 100, model = None) -> float:
    ds = InvoiceBatchDataset("data", split=split, min_n=5, max_n=15, size=n)
    accuracies = []
    chunk_scores = []
    for batch in tqdm(ds):
        pdf_path, y_true = batch
        y_pred = predict(pdf_path, model)
        scores = evaluate_predictions(y_true, y_pred)
        accuracies.append(scores["accuracy"])
        chunk_scores.append(scores["chunk_score"])
    accuracy = sum(accuracies) / len(accuracies)
    chunk_score = sum(chunk_scores) / len(chunk_scores)
    print(f"{split} results:")
    print(f"accuracy: {accuracy:.2f}")
    print(f"chunk score: {chunk_score:.2f}")
    return accuracy, chunk_score


if __name__ == "__main__":
    y_t = [1,0,0,1,0,0,1,0]
    y_p = [1,0,1,0,0,0,1,0]
    print(evaluate_predictions(y_t, y_p))
    