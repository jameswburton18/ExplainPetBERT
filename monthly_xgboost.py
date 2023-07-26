import pandas as pd
import torch
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
import argparse
from multi_modal_py.transformers_classifier import transformer_classifier
from multi_modal_py.metrics import get_Binary_predictions, get_Sigmoid_predictions
from xgboost import XGBClassifier
from multi_modal_py.tabular_data_encoder import auto_numpy_encode
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import re
import glob
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--train",
    default="Datasets/Modifiers/time_frames/1_month/masked/train.csv",
    help="path to train data",
)
parser.add_argument(
    "--test",
    default="Datasets/Modifiers/time_frames/1_month/masked/test.csv",
    help="path to test data",
)
parser.add_argument(
    "--mlm_model",
    default="biobert-v1.1-finetuned-Vet/checkpoint-625000",
    help="path to pretrained language model",
)
parser.add_argument("--epochs", default=100, help="number of epochs to train for")
parser.add_argument("--batch_size", default=16, help="batch size")
parser.add_argument(
    "--save_file",
    default="Datasets/Modifiers/time_frames/1_month/",
    help="path to save file",
)
parser.add_argument(
    "--classifier_model", default=None, help="path to pretrained classifier model"
)
args = vars(parser.parse_args())


df_train = pd.read_csv("all_datasets/train_concat.csv")
df_test = pd.read_csv("all_datasets/test_concat.csv")


df_train["Difference"] = df_train["Difference"].astype(str)
df_train["Difference"] = df_train["Difference"].str.split(" ").str[0]
df_train["Difference"] = df_train["Difference"].str.replace("nan", "999")
df_train["Difference"] = df_train["Difference"].astype(int)

df_test["Difference"] = df_test["Difference"].astype(str)
df_test["Difference"] = df_test["Difference"].str.split(" ").str[0]
df_test["Difference"] = df_test["Difference"].str.replace("nan", "999")
df_test["Difference"] = df_test["Difference"].astype(int)


Categorical_features = [
    "species",
    "breed",
    "practice_id",
    "premise_id",
    "gender",
    "neutered",
    "insured",
]

Numerical_features = [
    "age_at_consult",
    "Diseases of the ear or mastoid process",
    "Mental, behavioural or neurodevelopmental disorders",
    "Diseases of the blood or blood-forming organs",
    "Diseases of the circulatory system",
    "Dental",
    "Developmental anomalies",
    "Diseases of the digestive system",
    "Endocrine, nutritional or metabolic diseases",
    "Diseases of the Immune system",
    "Certain infectious or parasitic diseases",
    "Diseases of the skin",
    "Diseases of the musculoskeletal system or connective tissue",
    "Neoplasms",
    "Diseases of the nervous system",
    "Diseases of the visual system",
    "Certain conditions originating in the perinatal period",
    "Pregnancy, childbirth or the puerperium",
    "Diseases of the respiratory system",
    "Injury, poisoning or certain other consequences of external causes",
    "Diseases of the genitourinary system",
]

df_train = df_train[
    Categorical_features + Numerical_features + ["text"] + ["Difference"] + ["labels"]
]
df_test = df_test[
    Categorical_features + Numerical_features + ["text"] + ["Difference"] + ["labels"]
]

mask = r"(euth*|pts|pento*|died|dead|killed|death|put to sleep|crem|casket|decease|ashes|burial|qol|quality of life)"

df_train["text"] = df_train["text"].str.replace(
    mask, "[MASK]", regex=True, flags=re.IGNORECASE
)
df_train["text"] = df_train["text"].str.replace(
    r"\b(\w+)\b",
    lambda m: m.group(1) if np.random.rand() > 0.05 else "[MASK]",
    regex=True,
)


def get_latest_file(dir):
    list_of_files = glob.glob(
        dir + "/checkpoint-*"
    )  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


for month, i in enumerate(range(28, 28, 28)):
    # get file start within "checkpoint-*" in all_datasets/bal_monthly_results/month_0
    classifier_file = get_latest_file(
        "all_datasets/un_bal_monthly_results/month_" + str(month) + "/masked/"
    )
    save_file = "all_datasets/un_bal_monthly_results/month_" + str(month)

    df_train_cases = df_train[
        (df_train["Difference"] >= i) & (df_train["Difference"] < i + 28)
    ]
    df_test_cases = df_test[
        (df_test["Difference"] >= i) & (df_test["Difference"] < i + 28)
    ]

    df_train_controls = df_train[df_train["Difference"] == 999].sample(
        len(df_train_cases)
    )
    df_test_controls = df_test[df_test["Difference"] == 999].sample(len(df_test_cases))

    df_train_sample = pd.concat([df_train_cases, df_train_controls]).dropna()
    df_test_sample = pd.concat([df_test_cases, df_test_controls]).dropna()

    df_train_sample.data, df_train_label = auto_numpy_encode(
        df_train_sample, Categorical_features, Numerical_features
    )
    df_test_sample.data, df_test_label = auto_numpy_encode(
        df_test_sample, Categorical_features, Numerical_features
    )

    # XGBoost model
    xgb = XGBClassifier(tree_method="gpu_hist", gpu_id=0)
    xgb.fit((df_train_sample.data), df_train_label)

    train_xgb_Sigmoid_predictions = get_Sigmoid_predictions(
        xgb,
        df_train_sample.data,
    )
    train_xgb_Binary_predictions = get_Binary_predictions(
        df_train_sample.data, model=xgb
    )

    test_xgb_Sigmoid_predictions = get_Sigmoid_predictions(xgb, df_test_sample.data)
    test_xgb_Binary_predictions = get_Binary_predictions(df_test_sample.data, model=xgb)

    # Double training and testing sets so that final meta model can be trained on an unseen dataset
    train_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(df_train_sample),
            "test": Dataset.from_pandas(df_train_sample),
        }
    )
    test_datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(df_test_sample),
            "test": Dataset.from_pandas(df_test_sample),
        }
    )
    train_predictions, save_file = transformer_classifier(
        train_datasets,
        epochs=None,
        batch_size=16,
        mlm_model="biobert-v1.1-finetuned-Vet/checkpoint-625000",
        base_model="dmis-lab/biobert-v1.1",
        save_file=save_file + "/masked/",
        classifier_model=classifier_file,
        train=False,
        predictions=True,
    )

    test_predictions, _ = transformer_classifier(
        test_datasets,
        epochs=5,
        batch_size=16,
        mlm_model="biobert-v1.1-finetuned-Vet/checkpoint-625000",
        base_model="dmis-lab/biobert-v1.1",
        save_file=save_file,
        classifier_model=classifier_file,
        train=False,
        predictions=True,
    )

    train_transformer_Sigmoid_predictions = [
        torch.sigmoid(torch.tensor(probs))[1].item()
        for probs in train_predictions.predictions
    ]
    train_transformer_Binary_predictions = get_Binary_predictions(
        data=df_train_sample.data, predictions=train_transformer_Sigmoid_predictions
    )

    test_transformer_Sigmoid_predictions = [
        torch.sigmoid(torch.tensor(probs))[1].item()
        for probs in test_predictions.predictions
    ]
    test_transformer_Binary_predictions = get_Binary_predictions(
        data=df_test_sample.data, predictions=test_transformer_Sigmoid_predictions
    )

    # concatenate the predictions from XGBoost and transformer models to create a new feature matrix
    train_predictions_combined = np.column_stack(
        (
            train_xgb_Sigmoid_predictions,
            torch.tensor(train_transformer_Sigmoid_predictions),
        )
    )
    test_Predictions_combined = np.column_stack(
        (
            test_xgb_Sigmoid_predictions,
            torch.tensor(test_transformer_Sigmoid_predictions),
        )
    )

    # fit a meta-model on the concatenated predictions
    meta_model = RandomForestClassifier()
    meta_model.fit(train_predictions_combined, df_train_label)

    meta_model_predictions = get_Binary_predictions(
        test_Predictions_combined, model=meta_model
    )

    df_xgboost = (
        pd.DataFrame(
            classification_report(
                df_test_label, test_xgb_Binary_predictions, output_dict=True
            )
        )
        .transpose()
        .to_csv(
            "all_datasets/un_bal_monthly_results/month_"
            + str(month)
            + "/masked_new_xgboost_report.csv"
        )
    )
    df_transformer = (
        pd.DataFrame(
            classification_report(
                df_test_label, test_transformer_Binary_predictions, output_dict=True
            )
        )
        .transpose()
        .to_csv(
            "all_datasets/un_bal_monthly_results/month_"
            + str(month)
            + "/masked_transformer_report.csv"
        )
    )
    df_combined = (
        pd.DataFrame(
            classification_report(
                df_test_label, meta_model_predictions, output_dict=True
            )
        )
        .transpose()
        .to_csv(
            "all_datasets/un_bal_monthly_results/month_"
            + str(month)
            + "/masked_ensemble_report.csv"
        )
    )
    break
