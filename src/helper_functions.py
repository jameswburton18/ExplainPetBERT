import yaml
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)


class Config:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path, "r") as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)
            for key, value in yaml_data.items():
                setattr(self, key, value)


def row_to_string(row, cols):
    row["text"] = " | ".join(f"{col}: {row[col]}" for col in cols)
    return row


def prepare_text(dataset, di, version):
    """This is all for preparing the text part of the dataset
    Could be made more robust by referring to dataset_info.py instead"""

    if version == "all_text":
        cols = di.categorical_cols + di.numerical_cols + di.text_cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "text_col_only":
        cols = di.text_cols
        dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
        return dataset
    elif version == "record_only":
        # dataset rename column
        dataset = dataset.rename_column(di.text_cols[-1], "text")
        return dataset

    # elif version == "all_as_text_base_reorder":
    #     cols = di.base_reorder_cols[model_code]
    #     cols = cols[::-1] if reverse else cols
    #     dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
    #     return dataset
    # elif version == "all_as_text_tnt_reorder":
    #     cols = di.tnt_reorder_cols[model_code]
    #     cols = cols[::-1] if reverse else cols
    #     dataset = dataset.map(row_to_string, fn_kwargs={"cols": cols})
    #     return dataset

    else:
        raise ValueError(f"Unknown dataset type version ({version}) combination")


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
