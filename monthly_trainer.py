import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pandas as pd
import torch
from sklearn.metrics import classification_report
from datasets import Dataset, DatasetDict
import argparse
from multi_modal_py.transformers_classifier import transformer_classifier
from multi_modal_py.metrics import get_Binary_predictions
import glob


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    "--train", default="all_datasets/train_concat.csv",
    help="path to train data")
parser.add_argument(
    "--test", default="all_datasets/train_concat.csv",
    help="path to test data")
parser.add_argument(
    "--mlm_model", default="biobert-v1.1-finetuned-Vet/checkpoint-625000",
    help="path to pretrained language model")
parser.add_argument(
    "--epochs", default=100, help="number of epochs to train for")
parser.add_argument(
    "--batch_size", default=16, help="batch size")
parser.add_argument(
    "--save_file", default="Datasets/Modifiers/time_frames/1_month/", help="path to save file")
parser.add_argument(
    "--classifier_model", default="all_datasets/un_bal_monthly_results/month_0/checkpoint-4096",help="path to pretrained classifier model")
args = vars(parser.parse_args())


df_train = pd.read_csv(args['train'])
df_test = pd.read_csv(args['test'])

#Filter on age (young: 0-2, middle: 2-10, senior: 10+)
df_train = df_train[(df_train['age_at_consult'] > 10)]
df_test = df_test[(df_test['age_at_consult'] > 10)]

df_train = df_train[['text', 'Difference', 'labels']]
df_test = df_test[['text', 'Difference', 'labels']]

df_train['Difference'] = df_train['Difference'].astype(str)
df_train['Difference'] = df_train['Difference'].str.split(' ').str[0]
df_train['Difference'] = df_train['Difference'].str.replace('nan', "999")
df_train['Difference'] = df_train['Difference'].astype(int)

df_test['Difference'] = df_test['Difference'].astype(str)
df_test['Difference'] = df_test['Difference'].str.split(' ').str[0]
df_test['Difference'] = df_test['Difference'].str.replace('nan', "999")
df_test['Difference'] = df_test['Difference'].astype(int)


def get_latest_file(dir):
    list_of_files = glob.glob(dir + '/checkpoint-*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


for month, i in enumerate(range(28,364,28)):
    if i == 28:
        month+1
    save_file = "all_datasets/un_bal_monthly_results/month_"+str(month)
    
    df_train_cases = df_train[(df_train['Difference'] >= i) & (df_train['Difference'] < i+28)]
    df_test_cases = df_test[(df_test['Difference'] >= i) & (df_test['Difference'] < i+28)]
    

    df_train_controls = df_train[df_train['Difference'] == 999].sample(len(df_train_cases))
    df_test_controls = df_test[df_test['Difference'] == 999].sample(len(df_test_cases))
    
    df_train_sample = pd.concat([df_train_cases, df_train_controls]).dropna()
    df_test_sample = pd.concat([df_test_cases, df_test_controls]).dropna()
    

    datasets = DatasetDict(
        {"train": Dataset.from_pandas(df_train_sample), "test": Dataset.from_pandas(df_test_sample)}
    )

    if args['classifier_model'] == None:
        train_model = True
    else:
        train_model = False
        classifier_model_path = get_latest_file(save_file)
        
    train_modal_predictions, _ = transformer_classifier(datasets,
                                                    epochs=args["epochs"],
                                                    batch_size=args["batch_size"],
                                                    mlm_model=args["mlm_model"],
                                                    base_model="dmis-lab/biobert-v1.1",
                                                    classifier_model=classifier_model_path,
                                                    save_file=str(save_file),
                                                    train=train_model,
                                                    predictions=True,
                                                )

    transformer_Sigmoid_predictions = [
        torch.sigmoid(torch.tensor(probs))[1].item() for probs in train_modal_predictions.predictions
    ]
    transformer_Binary_predictions = get_Binary_predictions(
        data=df_test_sample["text"], predictions=transformer_Sigmoid_predictions
    )

    df_transformer = (
        pd.DataFrame(
            classification_report(
                df_test_sample["labels"], transformer_Binary_predictions, output_dict=True
            )
        )
        .transpose()
        .to_csv(str(save_file+"/senior_report.csv"))
    )
