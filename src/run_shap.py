import numpy as np
import shap
import pickle
from datasets import load_dataset

# from src.dataset_info import get_dataset_info
from transformers import pipeline, AutoTokenizer
import os
from tqdm import tqdm
from src.utils import token_segments, text_ft_index_ends, format_text_pred

# from src.models import Model
import xgboost as xgb
import lightgbm as lgb
from src.models import WeightedEnsemble, StackModel, AllAsTextModel
from src.joint_masker import JointMasker
from src.utils import ConfigLoader
import argparse
import scipy as sp


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="vet_10c_2_all_text",
    help="Name of config from the the multi_config.yaml file",
)


def run_shap(
    config_type,
    max_samples=100,
    test_set_size=100,
):
    # Shap args
    args = ConfigLoader(
        config_type, "configs/shap_configs.yaml", "configs/dataset_default.yaml"
    )
    # Data
    all_text_versions = [
        "all_text",
        "all_as_text",
        "all_as_text_base_reorder",
        "all_as_text_tnt_reorder",
    ]
    ds_name = args.ds_name
    train_df = load_dataset(
        ds_name,
        split="train",  # download_mode="force_redownload"
    ).to_pandas()
    y_train = train_df[args.label_col]

    test_df = load_dataset(
        ds_name,
        split="test",  # download_mode="force_redownload"
    ).to_pandas()
    test_df = test_df.sample(test_set_size, random_state=55)

    # Models
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model_base, model_max_length=512
    )
    if args.model_type in all_text_versions:
        text_pipeline = pipeline(
            "text-classification",
            model=args.my_text_model,
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            top_k=None,
        )
        # Define how to convert all columns to a single string
        if args.model_type in ["all_as_text", "all_text"]:
            cols_to_str_fn = lambda array: " | ".join(
                [
                    f"{col}: {val}"
                    for col, val in zip(
                        args.categorical_cols + args.numerical_cols + args.text_cols,
                        array,
                    )
                ]
            )
        else:
            # # Reorder based on the new index order in di
            # cols_to_str_fn = lambda array: " | ".join(
            #     [
            #         f"{col}: {val}"
            #         for _, col, val in sorted(
            #             zip(args.new_idx_order, args.tab_cols + args.text_cols, array)
            #         )
            #     ]
            # )
            raise NotImplementedError(
                "Shouldn't need much as the column ordering is in dataset info,\
                just need to update the cols_to_str_fn"
            )

        model = AllAsTextModel(
            text_pipeline=text_pipeline,
            cols_to_str_fn=cols_to_str_fn,
        )
    else:
        text_pipeline = pipeline(
            "text-classification",
            model=args.my_text_model,
            tokenizer=tokenizer,
            device="cuda:0",
            truncation=True,
            padding=True,
            top_k=None,
        )
        # Define how to convert the text columns to a single string
        if len(args.text_cols) == 1:

            def cols_to_str_fn(array):
                return array[0]

        else:

            def cols_to_str_fn(array):
                return " | ".join(
                    [f"{col}: {val}" for col, val in zip(args.text_cols, array)]
                )

        # LightGBM requires explicitly marking categorical features
        train_df[args.categorical_cols] = train_df[args.categorical_cols].astype(
            "category"
        )
        test_df[args.categorical_cols] = test_df[args.categorical_cols].astype(
            "category"
        )

        tab_model = lgb.LGBMClassifier(random_state=42, max_depth=3)
        tab_model.fit(train_df[args.categorical_cols + args.numerical_cols], y_train)

        if args.model_type in ["ensemble_25", "ensemble_50", "ensemble_75"]:
            text_weight = float(args.model_type.split("_")[-1]) / 100
            model = WeightedEnsemble(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                text_weight=text_weight,
                cols_to_str_fn=cols_to_str_fn,
            )
        elif args.model_type == "stack":
            """
            For the stack model, we make predictions on the validation set. These predictions
            are then used as features for the stack model (another LightGBM model) along with
            the other tabular features. In doing so the stack model learns, depending on the
            tabular features, when to trust the tabular model and when to trust the text model.
            """
            val_df = load_dataset(
                ds_name,
                split="validation",  # download_mode="force_redownload"
            ).to_pandas()
            val_df[args.categorical_cols] = val_df[args.categorical_cols].astype(
                "category"
            )
            y_val = val_df[args.label_col]
            val_text = list(map(cols_to_str_fn, val_df[args.text_cols].values))

            # Training set is the preditions from the tabular and text models on the validation set
            # plus the tabular features from the validation set
            text_val_preds = text_pipeline(val_text)
            text_val_preds = np.array(
                [format_text_pred(pred) for pred in text_val_preds]
            )
            # text_val_preds = np.array(
            #     [[lab["score"] for lab in pred] for pred in text_val_preds]
            # )

            # add text and tabular predictions to the val_df
            stack_val_df = val_df[args.categorical_cols + args.numerical_cols]
            tab_val_preds = tab_model.predict_proba(
                val_df[args.categorical_cols + args.numerical_cols]
            )
            stack_val_df[f"text_pred"] = text_val_preds[:, 1]
            stack_val_df[f"tab_pred"] = tab_val_preds[:, 1]

            stack_model = lgb.LGBMClassifier(
                random_state=42, max_depth=2, learning_rate=0.01
            )
            stack_model.fit(stack_val_df, y_val)

            model = StackModel(
                tab_model=tab_model,
                text_pipeline=text_pipeline,
                stack_model=stack_model,
                cols_to_str_fn=cols_to_str_fn,
            )
        else:
            raise ValueError(f"Invalid model type of {args.model_type}")

    np.random.seed(1)
    x = test_df[args.categorical_cols + args.numerical_cols + args.text_cols].values

    # We need to load the ordinal dataset so that we can calculate the correlations for
    # the masker
    ord_train_df = load_dataset(args.ord_ds_name, split="train").to_pandas()

    # Clustering only valid if there is more than one column
    if len(args.categorical_cols + args.numerical_cols) > 1:
        tab_pt = sp.cluster.hierarchy.complete(
            sp.spatial.distance.pdist(
                ord_train_df[args.categorical_cols + args.numerical_cols]
                .fillna(
                    ord_train_df[args.categorical_cols + args.numerical_cols].median()
                )
                .values.T,
                metric="correlation",
            )
        )
    else:
        tab_pt = None

    masker = JointMasker(
        tab_df=train_df[args.categorical_cols + args.numerical_cols],
        text_cols=args.text_cols,
        cols_to_str_fn=cols_to_str_fn,
        tokenizer=tokenizer,
        collapse_mask_token=True,
        max_samples=max_samples,
        tab_partition_tree=tab_pt,
    )

    explainer = shap.explainers.Partition(model=model.predict, masker=masker)
    shap_vals = explainer(x)

    output_dir = "models/shap_vals/"
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{config_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


def run_all_text_baseline_shap(
    config_type,
    test_set_size=100,
):
    # Shap args
    args = ConfigLoader(
        config_type, "configs/shap_configs.yaml", "configs/dataset_default.yaml"
    )
    # Data
    test_df = load_dataset(
        args.ds_name, split="test", download_mode="force_redownload"
    ).to_pandas()
    test_df = test_df.sample(test_set_size, random_state=55)

    # Models
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model_base, model_max_length=512
    )
    text_pipeline = pipeline(
        "text-classification",
        model=args.my_text_model,
        tokenizer=tokenizer,
        device="cuda:0",
        truncation=True,
        padding=True,
        top_k=None,
    )

    # Define how to convert all columns to a single string
    def cols_to_str_fn(array):
        return " | ".join(
            [
                f"{col}: {val}"
                for col, val in zip(
                    args.categorical_cols + args.numerical_cols + args.text_cols, array
                )
            ]
        )

    np.random.seed(1)
    x = list(
        map(
            cols_to_str_fn,
            test_df[
                args.categorical_cols + args.numerical_cols + args.text_cols
            ].values,
        )
    )
    explainer = shap.Explainer(text_pipeline, tokenizer)
    shap_vals = explainer(x)

    output_dir = "models/shap_vals/"
    print(f"Results will be saved @: {output_dir}")

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f"{config_type}.pkl"), "wb") as f:
        pickle.dump(shap_vals, f)

    return shap_vals


def load_shap_vals(config_name, add_parent_dir=False):
    pre = "../" if add_parent_dir else ""  # for running from notebooks
    with open(f"{pre}models/shap_vals/{config_name}.pkl", "rb") as f:
        shap_vals = pickle.load(f)
    return shap_vals


def gen_summary_shap_vals(config_type, add_parent_dir=False):
    # Shap args
    args = ConfigLoader(
        config_type, "configs/shap_configs.yaml", "configs/dataset_default.yaml"
    )
    shap_vals = load_shap_vals(config_type, add_parent_dir=add_parent_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model_base, model_max_length=512
    )
    filepath = f"models/shap_vals/summed_{config_type}.pkl"
    # filepath = f"models/shap_vals/summed_{config_type}_test.pkl"
    print(
        f"""
            #################
            {config_type}
            #################
            """
    )
    if "baseline" not in config_type:
        grouped_shap_vals = []
        for label in range(len(args.label_names)):
            shap_for_label = []
            for idx in tqdm(range(len(shap_vals))):
                sv = shap_vals[idx, :, label]
                text_ft_ends1 = text_ft_index_ends(
                    sv.data[len(args.categorical_cols + args.numerical_cols) :],
                    tokenizer,
                )
                text_ft_ends = [len(args.categorical_cols + args.numerical_cols)] + [
                    x + len(args.categorical_cols + args.numerical_cols) + 1
                    for x in text_ft_ends1
                ]
                val = np.append(
                    sv.values[: len(args.categorical_cols + args.numerical_cols)],
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                    + [np.sum(sv.values[text_ft_ends[-1] :])],
                )

                shap_for_label.append(val)
            grouped_shap_vals.append(np.vstack(shap_for_label))
        print(f"Saving to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(np.array(grouped_shap_vals), f)

    else:
        col_name_filepath = f"models/shap_vals/summed_{config_type}_col_names.pkl"
        colon_filepath = f"models/shap_vals/summed_{config_type}_colons.pkl"
        grouped_shap_vals = []
        grouped_col_name_shap_vals = []
        grouped_colon_shap_vals = []
        for label in range(len(args.label_names)):
            shap_for_label = []
            shap_for_col_name = []
            shap_for_colon = []
            for idx in tqdm(range(len(shap_vals))):
                sv = shap_vals[idx, :, label]
                stripped_data = np.array([item.strip() for item in sv.data])
                text_ft_ends = (
                    [1] + list(np.where(stripped_data == "|")[0]) + [len(sv.data) + 1]
                )
                # Need this if there are | in the text that aren't col separators
                # Not super robust and only implemented for the current col to text
                # mapping, but works for now
                if (
                    len(text_ft_ends)
                    != len(args.text_cols + args.categorical_cols + args.numerical_cols)
                    + 1
                ):
                    text_ft_ends = (
                        [1]
                        + [
                            i
                            for i in list(np.where(stripped_data == "|")[0])
                            if sv.data[i + 1].strip()
                            in [
                                token_segments(col, tokenizer)[0][1].strip()
                                for col in args.categorical_cols
                                + args.numerical_cols
                                + args.text_cols
                            ]
                            + args.categorical_cols
                            + args.numerical_cols
                            + args.text_cols
                        ]
                        + [len(sv.data) + 1]
                    )
                assert (
                    len(text_ft_ends)
                    == len(args.text_cols + args.categorical_cols + args.numerical_cols)
                    + 1
                )
                val = np.array(
                    [
                        np.sum(sv.values[text_ft_ends[i] : text_ft_ends[i + 1]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                )
                colon_idxs = np.where(stripped_data == ":")[0]
                col_idxs_after_ft = [
                    colon_idxs[list(np.where(colon_idxs > te)[0])[0]]
                    for te in text_ft_ends[:-1]
                ]
                ft_name_vals = np.array(
                    [
                        np.sum(sv.values[text_ft_ends[i] : col_idxs_after_ft[i]])
                        for i in range(len(text_ft_ends) - 1)
                    ]
                )
                colon_vals = np.array(sv.values[col_idxs_after_ft])
                shap_for_label.append(val)
                shap_for_col_name.append(ft_name_vals)
                shap_for_colon.append(colon_vals)
            grouped_shap_vals.append(np.vstack(shap_for_label))
            grouped_col_name_shap_vals.append(shap_for_col_name)
            grouped_colon_shap_vals.append(shap_for_colon)
        print(f"Saving to {filepath}")
        with open(filepath, "wb") as f:
            pickle.dump(np.array(grouped_shap_vals), f)
        print(f"Saving to {col_name_filepath}")
        with open(col_name_filepath, "wb") as f:
            pickle.dump(np.array(grouped_col_name_shap_vals), f)
        print(f"Saving to {colon_filepath}")
        with open(colon_filepath, "wb") as f:
            pickle.dump(np.array(grouped_colon_shap_vals), f)


if __name__ == "__main__":
    config_type = parser.parse_args().config
    # config_type = "vet_50c_all_text"
    if "baseline" in config_type:
        run_all_text_baseline_shap(config_type, test_set_size=1000)

    else:
        run_shap(config_type, test_set_size=1000)
    gen_summary_shap_vals(config_type)
