import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
import torch


def get_Sigmoid_predictions(model, data):
    predictions = model.predict_proba(data)[:,1]
    predictions_array = np.zeros(predictions.shape)
    predictions_array = torch.sigmoid(torch.tensor(predictions))
    return predictions_array

def get_Binary_predictions(data, threshold=0.5, model=None, predictions=None):
    if model != None:
        predictions = model.predict_proba(data)[:,1]
    else:
        predictions = np.array(predictions)
    predictions_array = np.zeros(predictions.shape)
    predictions_array[np.where(predictions >= threshold)] = 1
    return predictions_array

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
