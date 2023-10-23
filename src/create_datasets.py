from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from src.utils import ConfigLoader
from datasets import Dataset, DatasetDict

# OLD DOGS ONLY
####################################################################################################
# ## Dataset creation here
# for month in [
#     1,
# ]:
#     i = 28 * month
#     di = ConfigLoader("default", "", "configs/dataset_default.yaml")  # changed to 2
#     df_train = pd.read_csv("data/raw/train.csv")
#     df_test = pd.read_csv("data/raw/test.csv")

#     # rename each of di.numerical_cols_long to di.numerical_cols
#     for long_col, col in zip(di.numerical_cols_long, di.numerical_cols):
#         df_train = df_train.rename(columns={long_col: col})
#         df_test = df_test.rename(columns={long_col: col})
#     df_train = df_train.rename(columns={"text": "record"})
#     df_test = df_test.rename(columns={"text": "record"})
#     # Filter on age (young: 0-2, middle: 2-10, senior: 10+)
#     df_train = df_train[(df_train["age_at_consult"] > 10)]
#     df_test = df_test[(df_test["age_at_consult"] > 10)]
#     # Set numerical cols as int (apart from age_at_consult)
#     int_cols = di.numerical_cols[1:] + ["practice_id", "premise_id"]
#     df_train[int_cols] = df_train[int_cols].astype(int)
#     df_test[int_cols] = df_test[int_cols].astype(int)

#     df_train["Difference"] = df_train["Difference"].astype(str)
#     df_train["Difference"] = df_train["Difference"].str.split(" ").str[0]
#     df_train["Difference"] = df_train["Difference"].str.replace("nan", "999")
#     df_train["Difference"] = df_train["Difference"].astype(int)

#     df_test["Difference"] = df_test["Difference"].astype(str)
#     df_test["Difference"] = df_test["Difference"].str.split(" ").str[0]
#     df_test["Difference"] = df_test["Difference"].str.replace("nan", "999")
#     df_test["Difference"] = df_test["Difference"].astype(int)

#     df_train_cases = df_train[
#         (df_train["Difference"] >= i) & (df_train["Difference"] < i + 28)
#     ]
#     df_test_cases = df_test[
#         (df_test["Difference"] >= i) & (df_test["Difference"] < i + 28)
#     ]

#     df_train_controls = df_train[df_train["Difference"] == 999].sample(
#         len(df_train_cases)
#     )
#     df_test_controls = df_test[df_test["Difference"] == 999].sample(len(df_test_cases))

#     df_train_sample = pd.concat([df_train_cases, df_train_controls]).dropna()
#     df_test_sample = pd.concat([df_test_cases, df_test_controls]).dropna()

#     # drop columns that are not in di.text_cols, di.numerical_cols, di.categorical_cols
#     all_cols = di.numerical_cols + di.categorical_cols + di.text_cols + [di.label_col]
#     df_train_sample = df_train_sample[all_cols]
#     df_test_sample = df_test_sample[all_cols]

#     train_ds = Dataset.from_pandas(df_train_sample, preserve_index=False)
#     train_ds = train_ds.class_encode_column(di.label_col)
#     test_ds = Dataset.from_pandas(df_test_sample, preserve_index=False)
#     test_ds = test_ds.class_encode_column(di.label_col)
#     train_ds = train_ds.train_test_split(
#         test_size=0.15, seed=42, stratify_by_column=di.label_col
#     )

#     ds = DatasetDict(
#         {"train": train_ds["train"], "validation": train_ds["test"], "test": test_ds}
#     )

#     # Now we have made the split but still need to deal with missing values, and that
#     # depends on the column type

#     # All as text
#     ft_cols = di.numerical_cols + di.categorical_cols + di.text_cols
#     train_all_text = ds["train"].to_pandas()
#     train_all_text[ft_cols] = train_all_text[ft_cols].astype("str")
#     val_all_text = ds["validation"].to_pandas()
#     val_all_text[ft_cols] = val_all_text[ft_cols].astype("str")
#     test_all_text = ds["test"].to_pandas()
#     test_all_text[ft_cols] = test_all_text[ft_cols].astype("str")

#     ds_all_text = DatasetDict(
#         {
#             "train": Dataset.from_pandas(train_all_text, preserve_index=False),
#             "validation": Dataset.from_pandas(val_all_text, preserve_index=False),
#             "test": Dataset.from_pandas(test_all_text, preserve_index=False),
#         }
#     )

#     ds_all_text.push_to_hub(f"vet_month_{month}c_all_text")

#     train = ds["train"].to_pandas()
#     val = ds["validation"].to_pandas()
#     test = ds["test"].to_pandas()

#     train[di.text_cols] = train[di.text_cols].astype("str")
#     val[di.text_cols] = val[di.text_cols].astype("str")
#     test[di.text_cols] = test[di.text_cols].astype("str")

#     # ds.push_to_hub(dataset_name)
#     if len(di.categorical_cols) > 0:
#         train[di.categorical_cols] = train[di.categorical_cols].astype("category")

#         enc = OrdinalEncoder(
#             encoded_missing_value=-1,
#             handle_unknown="use_encoded_value",
#             unknown_value=-1,
#         )
#         train[di.categorical_cols] = enc.fit_transform(train[di.categorical_cols])

#         val[di.categorical_cols] = val[di.categorical_cols].astype("category")
#         val[di.categorical_cols] = enc.transform(val[di.categorical_cols])

#         test[di.categorical_cols] = test[di.categorical_cols].astype("category")
#         test[di.categorical_cols] = enc.transform(test[di.categorical_cols])

#     ds2 = DatasetDict(
#         {
#             "train": Dataset.from_pandas(train),
#             "validation": Dataset.from_pandas(val),
#             "test": Dataset.from_pandas(test),
#         }
#     )

#     ds2.push_to_hub(f"vet_month_{month}c_ordinal")

####################################################################################################
for month in [
    1,
]:
    i = 28 * month
    di = ConfigLoader("default", "", "configs/dataset_default.yaml")  # changed to 2
    df_train = pd.read_csv("data/raw/train.csv")
    df_test = pd.read_csv("data/raw/test.csv")

    # rename each of di.numerical_cols_long to di.numerical_cols
    for long_col, col in zip(di.numerical_cols_long, di.numerical_cols):
        df_train = df_train.rename(columns={long_col: col})
        df_test = df_test.rename(columns={long_col: col})
    df_train = df_train.rename(columns={"text": "record"})
    df_test = df_test.rename(columns={"text": "record"})
    # Filter on age (young: 0-2, middle: 2-10, senior: 10+)
    # df_train = df_train[(df_train["age_at_consult"] > 10)]
    # df_test = df_test[(df_test["age_at_consult"] > 10)]
    # Set numerical cols as int (apart from age_at_consult)
    int_cols = di.numerical_cols[1:] + ["practice_id", "premise_id"]
    df_train[int_cols] = df_train[int_cols].astype(int)
    df_test[int_cols] = df_test[int_cols].astype(int)

    df_train["Difference"] = df_train["Difference"].astype(str)
    df_train["Difference"] = df_train["Difference"].str.split(" ").str[0]
    df_train["Difference"] = df_train["Difference"].str.replace("nan", "999")
    df_train["Difference"] = df_train["Difference"].astype(int)

    df_test["Difference"] = df_test["Difference"].astype(str)
    df_test["Difference"] = df_test["Difference"].str.split(" ").str[0]
    df_test["Difference"] = df_test["Difference"].str.replace("nan", "999")
    df_test["Difference"] = df_test["Difference"].astype(int)

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

    # drop columns that are not in di.text_cols, di.numerical_cols, di.categorical_cols
    all_cols = di.numerical_cols + di.categorical_cols + di.text_cols + [di.label_col]
    df_train_sample = df_train_sample[all_cols]
    df_test_sample = df_test_sample[all_cols]

    train_ds = Dataset.from_pandas(df_train_sample, preserve_index=False)
    train_ds = train_ds.class_encode_column(di.label_col)
    test_ds = Dataset.from_pandas(df_test_sample, preserve_index=False)
    test_ds = test_ds.class_encode_column(di.label_col)
    train_ds = train_ds.train_test_split(
        test_size=0.15, seed=42, stratify_by_column=di.label_col
    )

    ds = DatasetDict(
        {"train": train_ds["train"], "validation": train_ds["test"], "test": test_ds}
    )

    # Now we have made the split but still need to deal with missing values, and that
    # depends on the column type

    # All as text
    ft_cols = di.numerical_cols + di.categorical_cols + di.text_cols
    train_all_text = ds["train"].to_pandas()
    train_all_text[ft_cols] = train_all_text[ft_cols].astype("str")
    val_all_text = ds["validation"].to_pandas()
    val_all_text[ft_cols] = val_all_text[ft_cols].astype("str")
    test_all_text = ds["test"].to_pandas()
    test_all_text[ft_cols] = test_all_text[ft_cols].astype("str")

    ds_all_text = DatasetDict(
        {
            "train": Dataset.from_pandas(train_all_text, preserve_index=False),
            "validation": Dataset.from_pandas(val_all_text, preserve_index=False),
            "test": Dataset.from_pandas(test_all_text, preserve_index=False),
        }
    )

    ds_all_text.push_to_hub(f"vet_month_{month}d_all_text")

    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()

    train[di.text_cols] = train[di.text_cols].astype("str")
    val[di.text_cols] = val[di.text_cols].astype("str")
    test[di.text_cols] = test[di.text_cols].astype("str")

    # ds.push_to_hub(dataset_name)
    if len(di.categorical_cols) > 0:
        train[di.categorical_cols] = train[di.categorical_cols].astype("category")

        enc = OrdinalEncoder(
            encoded_missing_value=-1,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        train[di.categorical_cols] = enc.fit_transform(train[di.categorical_cols])

        val[di.categorical_cols] = val[di.categorical_cols].astype("category")
        val[di.categorical_cols] = enc.transform(val[di.categorical_cols])

        test[di.categorical_cols] = test[di.categorical_cols].astype("category")
        test[di.categorical_cols] = enc.transform(test[di.categorical_cols])

    ds2 = DatasetDict(
        {
            "train": Dataset.from_pandas(train),
            "validation": Dataset.from_pandas(val),
            "test": Dataset.from_pandas(test),
        }
    )

    ds2.push_to_hub(f"vet_month_{month}d_ordinal")
