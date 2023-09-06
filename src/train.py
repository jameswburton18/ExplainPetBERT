from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_from_disk, load_dataset
import wandb
import pandas as pd
import os
import yaml
import argparse
from transformers.trainer_callback import EarlyStoppingCallback
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from datasets import Dataset, DatasetDict
from src.helper_functions import Config, prepare_text, compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="testing ",
    help="Name of config from the the multi_config.yaml file",
)
config_type = parser.parse_args().config


def main():
    # Import yaml file
    with open("configs/train_default.yaml") as f:
        args = yaml.safe_load(f)

    # Update default args with chosen config
    if config_type != "default":
        with open("configs/train_configs.yaml") as f:
            yaml_configs = yaml.safe_load_all(f)
            yaml_args = next(
                conf for conf in yaml_configs if conf["config"] == config_type
            )
        args.update(yaml_args)
        print(f"Updating with:\n{yaml_args}\n")
    print(f"\n{args}\n")

    # Dataset
    di = Config("configs/dataset_info.yaml")
    dataset = load_dataset(
        args["ds_name"],
        # download_mode="force_redownload",
    )
    dataset = prepare_text(
        dataset=dataset,
        di=di,
        version=args["version"],
    )

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        args["model_base"],
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.get("tokenizer_base", args["model_base"])
    )

    # Tokenize the dataset
    def encode(examples):
        return {
            "label": np.array([examples["labels"]]),
            **tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=args["max_length"],
            ),
        }

    # Fast dev run if want to run quickly and not save to wandb
    if args["fast_dev_run"]:
        args["num_epochs"] = 1
        args["tags"].append("fast-dev-run")
        dataset["train"] = dataset["train"].select(range(100))
        dataset["validation"] = dataset["validation"].select(range(50))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(50))
        output_dir = os.path.join(args["output_root"], "testing")
        print(
            "\n######################    Running in fast dev mode    #######################\n"
        )

    # If not, initialize wandb
    else:
        wandb.init(
            project="PetBERT",
            tags=args["tags"],
            save_code=True,
            config={"my_args/" + k: v for k, v in args.items()},
        )
        os.environ["WANDB_LOG_MODEL"] = "True"
        output_dir = os.path.join(args["output_root"], config_type, wandb.run.name)
        print(f"Results will be saved @: {output_dir}")

    dataset = dataset.map(encode)  # , load_from_cache_file=True)
    dataset = dataset.remove_columns(["labels"])

    # Make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save args file
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)

    # Initialise training arguments and trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args["num_epochs"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        logging_steps=args["logging_steps"],
        # learning_rate=args["lr"],
        # weight_decay=args["weight_decay"],
        # gradient_accumulation_steps=args["grad_accumulation_steps"],
        # warmup_ratio=args["warmup_ratio"],
        warmup_steps=args.get("warmup_steps", 0),
        weight_decay=args.get("weight_decay", 0),
        # lr_scheduler_type=args["lr_scheduler"],
        dataloader_num_workers=args["num_workers"],
        do_train=args["do_train"],
        do_predict=args["do_predict"],
        resume_from_checkpoint=args["resume_from_checkpoint"],
        report_to="wandb" if not args["fast_dev_run"] else "none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args["save_total_limit"],
        load_best_model_at_end=True,
        seed=args["seed"],
        torch_compile=args["pytorch2.0"],  # Needs to be true if PyTorch 2.0
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=[EarlyStoppingCallback(args["early_stopping_patience"])]
        if args["early_stopping_patience"] > 0
        else [],
    )

    # Train model
    if args["do_train"]:
        print("Training...")
        trainer.train()
        if not args["fast_dev_run"]:
            model.push_to_hub(config_type, private=True)
        print("Training complete")

    # Predict on the test set
    if args["do_predict"]:
        print("***** Running Prediction *****")
        # Test the model
        results = trainer.evaluate(dataset["test"], metric_key_prefix="test")
        preds = trainer.predict(dataset["test"]).predictions
        labels = [lab[0] for lab in dataset["test"]["label"]]
        results["test/accuracy"] = np.mean(np.argmax(preds, axis=1) == labels)
        results["test/precision"] = precision_score(
            labels,
            np.argmax(preds, axis=1),
            labels=np.arange(2),  # num_labels
            zero_division=0,
        )
        results["test/recall"] = recall_score(
            labels,
            np.argmax(preds, axis=1),
            labels=np.arange(2),  # num_labels
            zero_division=0,
        )
        results["test/roc_auc"] = roc_auc_score(labels, preds[:, 1])
        results["test/f1"] = (
            2
            * results["test/precision"]
            * results["test/recall"]
            / (results["test/precision"] + results["test/recall"])
        )

        # Save the predictions
        with open(os.path.join(output_dir, "test_results.txt"), "w") as f:
            f.write(str(results))
        if not args["fast_dev_run"]:
            wandb.log(results)

    print("Predictions complete")


if __name__ == "__main__":
    main()
