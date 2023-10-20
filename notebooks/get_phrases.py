import pickle
from transformers import AutoTokenizer
import numpy as np
import shap
from src.plot_text import text, get_grouped_vals

# from src.utils import format_fts_for_plotting, text_ft_index_ends, token_segments
# from src.utils import legacy_get_dataset_info
from datasets import load_dataset

# import matplotlib.pyplot as plt
from src.run_shap import load_shap_vals
from tqdm import tqdm

# import re
# import seaborn as sns
# from scipy import stats
from src.utils import (
    ConfigLoader,
    text_ft_index_ends,
    token_segments,
    format_fts_for_plotting,
    format_text_fts_too,
)
from src.get_phrases import get_phrases
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="vet_50c_all_text",
    help="Name of config from the the multi_config.yaml file",
)


def get_all_phrases(config_name, add_parent_dir=False):
    pre = "../" if add_parent_dir else ""
    args = ConfigLoader(
        config_name,
        f"{pre}configs/shap_configs.yaml",
        f"{pre}configs/dataset_default.yaml",
    )
    shap_vals = load_shap_vals(config_name, add_parent_dir=add_parent_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_base)
    all_tokens = np.array([])
    all_values = np.array([])
    for idx in tqdm(range(len(shap_vals))):
        # for idx in tqdm(range(10)):
        text_idxs = text_ft_index_ends(
            text_fts=shap_vals.data[idx][
                len(args.categorical_cols + args.numerical_cols) :
            ],
            tokenizer=tokenizer,
        )
        linebreak_before_idxs = [len(args.categorical_cols + args.numerical_cols)] + [
            x + len(args.categorical_cols + args.numerical_cols) + 1 for x in text_idxs
        ]

        formatted_data = np.array(
            format_fts_for_plotting(
                shap_vals[idx].feature_names,
                shap_vals[idx].data[: len(args.categorical_cols + args.numerical_cols)],
            )
        )

        formatted_data = format_text_fts_too(
            formatted_data,
            linebreak_before_idxs,
            args.text_cols,
        )
        tokens, values = get_phrases(
            shap.Explanation(
                values=shap_vals[idx].values,
                base_values=shap_vals[idx].base_values,
                data=formatted_data,
                clustering=shap_vals[idx].clustering,
                output_names=args.label_names,
                hierarchical_values=shap_vals[idx].hierarchical_values,
            ),
            # grouping_threshold=20,
            grouping_threshold=0.5,
        )
        all_tokens = np.concatenate((all_tokens, tokens))
        all_values = np.concatenate((all_values, values))
    return all_tokens, all_values


def get_tab_vals_back(config_name, add_parent_dir=False):
    pre = "../" if add_parent_dir else ""
    args = ConfigLoader(
        config_name,
        f"{pre}configs/shap_configs.yaml",
        f"{pre}configs/dataset_default.yaml",
    )
    shap_vals = load_shap_vals(config_name, add_parent_dir=add_parent_dir)

    # Make an empty array of len(shap_vals) by len(args.categorical_cols + args.numerical_cols)
    all_tab_tokens = np.empty(
        (len(shap_vals), len(args.categorical_cols + args.numerical_cols)), dtype=object
    )
    all_tab_values = np.empty(
        (len(shap_vals), len(args.categorical_cols + args.numerical_cols)), dtype=float
    )
    for idx in tqdm(range(len(shap_vals))):
        formatted_data = np.array(
            format_fts_for_plotting(
                shap_vals[idx].feature_names,
                shap_vals[idx].data[: len(args.categorical_cols + args.numerical_cols)],
            )
        )
        all_tab_tokens[idx] = formatted_data[
            : len(args.categorical_cols + args.numerical_cols)
        ]
        all_tab_values[idx] = shap_vals[
            idx, : len(args.categorical_cols + args.numerical_cols), 0
        ].values

    return all_tab_tokens, all_tab_values


if __name__ == "__main__":
    config_type = parser.parse_args().config
    # all_tokens, all_values = get_all_phrases(config_type)
    # # save to file
    # with open(f"{config_type}_all_tokens_50.pkl", "wb") as f:
    #     pickle.dump(all_tokens, f)
    # with open(f"{config_type}_all_values_50.pkl", "wb") as f:
    #     pickle.dump(all_values, f)
    tab_tokens, tab_values = get_tab_vals_back(config_type)
    with open(f"{config_type}_tab_tokens.pkl", "wb") as f:
        pickle.dump(tab_tokens, f)
    with open(f"{config_type}_tab_values.pkl", "wb") as f:
        pickle.dump(tab_values, f)
    # print(all_tokens)
