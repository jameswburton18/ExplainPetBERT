import yaml
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
import numpy as np
from shap.utils import safe_isinstance
from shap.utils.transformers import (
    parse_prefix_suffix_for_tokenizer,
    SENTENCEPIECE_TOKENIZERS,
    getattr_silent,
)
import json
import random
import string

import numpy as np
from shap.plots import colors
from shap.plots._text import (
    process_shap_values,
    unpack_shap_explanation_contents,
    svg_force_plot,
    # text as shap_text,
)

try:
    from IPython.display import HTML
    from IPython.display import display as ipython_display

    have_ipython = True
except ImportError:
    have_ipython = False


# from src.dataset_info import get_dataset_info


def format_text_pred(pred):
    scores = [p["score"] for p in pred]
    order = [int(p["label"][6:]) for p in pred]
    return np.array(
        [scores[i] for i in sorted(range(len(scores)), key=lambda x: order[x])]
    )


class ConfigLoader:
    def __init__(self, config_name, configs_path, default_path=None):
        if default_path is not None:
            with open(default_path) as f:
                args = yaml.safe_load(f)

        # Update default args with chosen config
        if config_name != "default":
            with open(configs_path) as f:
                yaml_configs = yaml.safe_load_all(f)
                try:
                    yaml_args = next(
                        conf for conf in yaml_configs if conf["config"] == config_name
                    )
                except StopIteration:
                    raise ValueError(
                        f"Config name {config_name} not found in {configs_path}"
                    )
            if default_path is not None:
                args.update(yaml_args)
                print(f"Updating with:\n{yaml_args}\n")
            else:
                args = yaml_args
        print(f"\n{args}\n")
        for key, value in args.items():
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


def compute_metrics(p, argmax=True):
    pred, labels = p
    pred = np.argmax(pred, axis=1) if argmax else pred
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def format_fts_for_plotting(fts, tab_data):
    for i in range(len(tab_data)):
        fts[i] = fts[i] + f" = {tab_data[i]}   "
    # for j in range(len(tab_data), len(fts)):
    #     fts[j] = fts[j] + ""
    return fts


def format_text_fts_too(formatted_data, linebreak_before_idxs, text_cols):
    cols_added = 0
    for i, data in enumerate(formatted_data):
        if i in linebreak_before_idxs:
            formatted_data[i] = f"(Text ft) {text_cols[cols_added]} = {data}   "
            cols_added += 1
    return formatted_data


def text_ft_index_ends(text_fts, tokenizer):
    lens = []
    sent_indices = []
    parsed_tokenizer_dict = parse_prefix_suffix_for_tokenizer(tokenizer)

    keep_prefix = parsed_tokenizer_dict["keep_prefix"]
    keep_suffix = parsed_tokenizer_dict["keep_suffix"]
    for idx, col in enumerate(text_fts):
        # First text col
        if lens == []:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 as we don't use SEP tokens (unless it's the only text col)
            also_last = 1 if len(text_fts) == 1 else 0
            token_len = len(tokens) - keep_suffix + also_last
            lens.append(token_len - 1)
            sent_indices.extend([idx] * token_len)
        # Last text col
        elif idx == len(text_fts) - 1:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -1 for CLS tokens
            token_len = len(tokens) - keep_prefix
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)
        # Middle text cols
        else:
            tokens, token_ids = token_segments(str(col), tokenizer)
            # -2 for CLS and SEP tokens
            token_len = len(tokens) - keep_prefix - keep_suffix
            lens.append(lens[-1] + token_len)
            sent_indices.extend([idx] * token_len)

    return lens[:-1]


def token_segments(s, tokenizer):
    """Same as Text masker"""
    """ Returns the substrings associated with each token in the given string.
    """

    try:
        token_data = tokenizer(s, return_offsets_mapping=True)
        offsets = token_data["offset_mapping"]
        offsets = [(0, 0) if o is None else o for o in offsets]
        parts = [
            s[offsets[i][0] : max(offsets[i][1], offsets[i + 1][0])]
            for i in range(len(offsets) - 1)
        ]
        parts.append(s[offsets[len(offsets) - 1][0] : offsets[len(offsets) - 1][1]])
        return parts, token_data["input_ids"]
    except (
        NotImplementedError,
        TypeError,
    ):  # catch lack of support for return_offsets_mapping
        token_ids = tokenizer(s)["input_ids"]
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
        else:
            tokens = [tokenizer.decode([id]) for id in token_ids]
        if hasattr(tokenizer, "get_special_tokens_mask"):
            special_tokens_mask = tokenizer.get_special_tokens_mask(
                token_ids, already_has_special_tokens=True
            )
            # avoid masking separator tokens, but still mask beginning of sentence and end of sentence tokens
            special_keep = [
                getattr_silent(tokenizer, "sep_token"),
                getattr_silent(tokenizer, "mask_token"),
            ]
            for i, v in enumerate(special_tokens_mask):
                if v == 1 and (
                    tokens[i] not in special_keep or i + 1 == len(special_tokens_mask)
                ):
                    tokens[i] = ""

        # add spaces to separate the tokens (since we want segments not tokens)
        if safe_isinstance(tokenizer, SENTENCEPIECE_TOKENIZERS):
            for i, v in enumerate(tokens):
                if v.startswith("_"):
                    tokens[i] = " " + tokens[i][1:]
        else:
            for i, v in enumerate(tokens):
                if v.startswith("##"):
                    tokens[i] = tokens[i][2:]
                elif v != "" and i != 0:
                    tokens[i] = " " + tokens[i]

        return tokens, token_ids


# # TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# # the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
# def text(
#     shap_values,
#     num_starting_labels=0,
#     grouping_threshold=0.01,
#     separator="",
#     xmin=None,
#     xmax=None,
#     cmax=None,
#     display=True,
# ):
#     """Plots an explanation of a string of text using coloring and interactive labels.

#     The output is interactive HTML and you can click on any token to toggle the display of the
#     SHAP value assigned to that token.

#     Parameters
#     ----------
#     shap_values : [numpy.array]
#         List of arrays of SHAP values. Each array has the shap values for a string (#input_tokens x output_tokens).

#     num_starting_labels : int
#         Number of tokens (sorted in descending order by corresponding SHAP values)
#         that are uncovered in the initial view.
#         When set to 0, all tokens are covered.

#     grouping_threshold : float
#         If the component substring effects are less than a ``grouping_threshold``
#         fraction of an unlowered interaction effect, then we visualize the entire group
#         as a single chunk. This is primarily used for explanations that were computed
#         with fixed_context set to 1 or 0 when using the :class:`.explainers.Partition`
#         explainer, since this causes interaction effects to be left on internal nodes
#         rather than lowered.

#     separator : string
#         The string separator that joins tokens grouped by interaction effects and
#         unbroken string spans. Defaults to the empty string ``""``.

#     xmin : float
#         Minimum shap value bound.

#     xmax : float
#         Maximum shap value bound.

#     cmax : float
#         Maximum absolute shap value for sample. Used for scaling colors for input tokens.

#     display: bool
#         Whether to display or return html to further manipulate or embed. Default: ``True``

#     Examples
#     --------

#     See `text plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/text.html>`_.

#     """

#     def values_min_max(values, base_values):
#         """Used to pick our axis limits."""
#         fx = base_values + values.sum()
#         xmin = fx - values[values > 0].sum()
#         xmax = fx - values[values < 0].sum()
#         cmax = max(abs(values.min()), abs(values.max()))
#         d = xmax - xmin
#         xmin -= 0.1 * d
#         xmax += 0.1 * d

#         return xmin, xmax, cmax

#     uuid = "".join(random.choices(string.ascii_lowercase, k=20))

#     # loop when we get multi-row inputs
#     if len(shap_values.shape) == 2 and (
#         shap_values.output_names is None or isinstance(shap_values.output_names, str)
#     ):
#         xmin = 0
#         xmax = 0
#         cmax = 0

#         for i, v in enumerate(shap_values):
#             values, clustering = unpack_shap_explanation_contents(v)
#             tokens, values, group_sizes = process_shap_values(
#                 v.data, values, grouping_threshold, separator, clustering
#             )

#             if i == 0:
#                 xmin, xmax, cmax = values_min_max(values, v.base_values)
#                 continue

#             xmin_i, xmax_i, cmax_i = values_min_max(values, v.base_values)
#             if xmin_i < xmin:
#                 xmin = xmin_i
#             if xmax_i > xmax:
#                 xmax = xmax_i
#             if cmax_i > cmax:
#                 cmax = cmax_i
#         out = ""
#         for i, v in enumerate(shap_values):
#             out += f"""
#     <br>
#     <hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
#     <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
#     </div>
#                 """
#             out += text(
#                 v,
#                 num_starting_labels=num_starting_labels,
#                 grouping_threshold=grouping_threshold,
#                 separator=separator,
#                 xmin=xmin,
#                 xmax=xmax,
#                 cmax=cmax,
#                 display=False,
#             )
#         if display:
#             return tokens, values
#             return
#         else:
#             return out

#     if len(shap_values.shape) == 2 and shap_values.output_names is not None:
#         xmin_computed = None
#         xmax_computed = None
#         cmax_computed = None

#         for i in range(shap_values.shape[-1]):
#             values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
#             tokens, values, group_sizes = process_shap_values(
#                 shap_values[:, i].data,
#                 values,
#                 grouping_threshold,
#                 separator,
#                 clustering,
#             )

#             # if i == 0:
#             #     xmin, xmax, cmax = values_min_max(values, shap_values[:,i].base_values)
#             #     continue

#             xmin_i, xmax_i, cmax_i = values_min_max(
#                 values, shap_values[:, i].base_values
#             )
#             if xmin_computed is None or xmin_i < xmin_computed:
#                 xmin_computed = xmin_i
#             if xmax_computed is None or xmax_i > xmax_computed:
#                 xmax_computed = xmax_i
#             if cmax_computed is None or cmax_i > cmax_computed:
#                 cmax_computed = cmax_i

#         if xmin is None:
#             xmin = xmin_computed
#         if xmax is None:
#             xmax = xmax_computed
#         if cmax is None:
#             cmax = cmax_computed

#         out = f"""<div align='center'>
# <script>
#     document._hover_{uuid} = '_tp_{uuid}_output_0';
#     document._zoom_{uuid} = undefined;
#     function _output_onclick_{uuid}(i) {{
#         var next_id = undefined;

#         if (document._zoom_{uuid} !== undefined) {{
#             document.getElementById(document._zoom_{uuid}+ '_zoom').style.display = 'none';

#             if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
#                 document.getElementById(document._zoom_{uuid}).style.display = 'block';
#                 document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = '3px solid #000000';
#             }} else {{
#                 document.getElementById(document._zoom_{uuid}).style.display = 'none';
#                 document.getElementById(document._zoom_{uuid}+'_name').style.borderBottom = 'none';
#             }}
#         }}
#         if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
#             next_id = '_tp_{uuid}_output_' + i;
#             document.getElementById(next_id).style.display = 'none';
#             document.getElementById(next_id + '_zoom').style.display = 'block';
#             document.getElementById(next_id+'_name').style.borderBottom = '3px solid #000000';
#         }}
#         document._zoom_{uuid} = next_id;
#     }}
#     function _output_onmouseover_{uuid}(i, el) {{
#         if (document._zoom_{uuid} !== undefined) {{ return; }}
#         if (document._hover_{uuid} !== undefined) {{
#             document.getElementById(document._hover_{uuid} + '_name').style.borderBottom = 'none';
#             document.getElementById(document._hover_{uuid}).style.display = 'none';
#         }}
#         document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
#         el.style.borderBottom = '3px solid #000000';
#         document._hover_{uuid} = '_tp_{uuid}_output_' + i;
#     }}
# </script>
# <div style=\"color: rgb(120,120,120); font-size: 12px;\">outputs</div>"""
#         output_values = shap_values.values.sum(0) + shap_values.base_values
#         output_max = np.max(np.abs(output_values))
#         for i, name in enumerate(shap_values.output_names):
#             scaled_value = 0.5 + 0.5 * output_values[i] / (output_max + 1e-8)
#             color = colors.red_transparent_blue(scaled_value)
#             color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])
#             # '#dddddd' if i == 0 else '#ffffff' border-bottom: {'3px solid #000000' if i == 0 else 'none'};
#             out += f"""
# <div style="display: inline; border-bottom: {'3px solid #000000' if i == 0 else 'none'}; background: rgba{color}; border-radius: 3px; padding: 0px" id="_tp_{uuid}_output_{i}_name"
#     onclick="_output_onclick_{uuid}({i})"
#     onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>"""
#         out += "<br><br>"
#         for i, name in enumerate(shap_values.output_names):
#             out += f"<div id='_tp_{uuid}_output_{i}' style='display: {'block' if i == 0 else 'none'}';>"
#             out += text(
#                 shap_values[:, i],
#                 num_starting_labels=num_starting_labels,
#                 grouping_threshold=grouping_threshold,
#                 separator=separator,
#                 xmin=xmin,
#                 xmax=xmax,
#                 cmax=cmax,
#                 display=False,
#             )
#             out += "</div>"
#             out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
#             out += text(
#                 shap_values[:, i],
#                 num_starting_labels=num_starting_labels,
#                 grouping_threshold=grouping_threshold,
#                 separator=separator,
#                 display=False,
#             )
#             out += "</div>"
#         out += "</div>"
#         if display:
#             return tokens, values
#         else:
#             return out
#         # text_to_text(shap_values)
#         # return

#     if len(shap_values.shape) == 3:
#         xmin_computed = None
#         xmax_computed = None
#         cmax_computed = None

#         for i in range(shap_values.shape[-1]):
#             for j in range(shap_values.shape[0]):
#                 values, clustering = unpack_shap_explanation_contents(
#                     shap_values[j, :, i]
#                 )
#                 tokens, values, group_sizes = process_shap_values(
#                     shap_values[j, :, i].data,
#                     values,
#                     grouping_threshold,
#                     separator,
#                     clustering,
#                 )

#                 xmin_i, xmax_i, cmax_i = values_min_max(
#                     values, shap_values[j, :, i].base_values
#                 )
#                 if xmin_computed is None or xmin_i < xmin_computed:
#                     xmin_computed = xmin_i
#                 if xmax_computed is None or xmax_i > xmax_computed:
#                     xmax_computed = xmax_i
#                 if cmax_computed is None or cmax_i > cmax_computed:
#                     cmax_computed = cmax_i

#         if xmin is None:
#             xmin = xmin_computed
#         if xmax is None:
#             xmax = xmax_computed
#         if cmax is None:
#             cmax = cmax_computed

#         out = ""
#         for i, v in enumerate(shap_values):
#             out += f"""
# <br>
# <hr style="height: 1px; background-color: #fff; border: none; margin-top: 18px; margin-bottom: 18px; border-top: 1px dashed #ccc;"">
# <div align="center" style="margin-top: -35px;"><div style="display: inline-block; background: #fff; padding: 5px; color: #999; font-family: monospace">[{i}]</div>
# </div>
#             """
#             out += text(
#                 v,
#                 num_starting_labels=num_starting_labels,
#                 grouping_threshold=grouping_threshold,
#                 separator=separator,
#                 xmin=xmin,
#                 xmax=xmax,
#                 cmax=cmax,
#                 display=False,
#             )
#         if display:
#             return tokens, values
#         else:
#             return out

#     # set any unset bounds
#     xmin_new, xmax_new, cmax_new = values_min_max(
#         shap_values.values, shap_values.base_values
#     )
#     if xmin is None:
#         xmin = xmin_new
#     if xmax is None:
#         xmax = xmax_new
#     if cmax is None:
#         cmax = cmax_new

#     values, clustering = unpack_shap_explanation_contents(shap_values)
#     tokens, values, group_sizes = process_shap_values(
#         shap_values.data, values, grouping_threshold, separator, clustering
#     )

#     # build out HTML output one word one at a time
#     top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
#     out = ""
#     # ev_str = str(shap_values.base_values)
#     # vsum_str = str(values.sum())
#     # fx_str = str(shap_values.base_values + values.sum())

#     # uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
#     encoded_tokens = [
#         t.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "") for t in tokens
#     ]
#     output_name = (
#         shap_values.output_names if isinstance(shap_values.output_names, str) else ""
#     )
#     out += svg_force_plot(
#         values,
#         shap_values.base_values,
#         shap_values.base_values + values.sum(),
#         encoded_tokens,
#         uuid,
#         xmin,
#         xmax,
#         output_name,
#     )
#     out += "<div align='center'><div style=\"color: rgb(120,120,120); font-size: 12px; margin-top: -15px;\">inputs</div>"
#     for i, token in enumerate(tokens):
#         scaled_value = 0.5 + 0.5 * values[i] / (cmax + 1e-8)
#         color = colors.red_transparent_blue(scaled_value)
#         color = (color[0] * 255, color[1] * 255, color[2] * 255, color[3])

#         # display the labels for the most important words
#         label_display = "none"
#         wrapper_display = "inline"
#         if i in top_inds:
#             label_display = "block"
#             wrapper_display = "inline-block"

#         # create the value_label string
#         value_label = ""
#         if group_sizes[i] == 1:
#             value_label = str(values[i].round(3))
#         else:
#             value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])

#         # the HTML for this token
#         out += f"""<div style='display: {wrapper_display}; text-align: center;'
#     ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: 12px;'>{value_label}</div
#         ><div id='_tp_{uuid}_ind_{i}'
#             style='display: inline; background: rgba{color}; border-radius: 3px; padding: 0px'
#             onclick="
#             if (this.previousSibling.style.display == 'none') {{
#                 this.previousSibling.style.display = 'block';
#                 this.parentNode.style.display = 'inline-block';
#             }} else {{
#                 this.previousSibling.style.display = 'none';
#                 this.parentNode.style.display = 'inline';
#             }}"
#             onmouseover="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
#             onmouseout="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
#         >{token.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '')}</div></div>"""
#     out += "</div>"

#     if display:
#         # return tokens, values
#         print("here")
#         return tokens, values
#     else:
#         return out
