from datasets import Dataset, DatasetDict, Features, Value
import numpy as np
import numpy as np
import pandas as pd
import glob
import os

from transformers import (
    AutoConfig,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from multi_modal_py.metrics import compute_metrics


def transformer_classifier(
    datasets, batch_size, mlm_model, base_model, save_file, epochs=None,classifier_model=None,  train=True, predictions=False):
    print(f'save_file: {save_file}')
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if epochs == None:
        epochs = 100
        
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], padding=True, truncation=True, max_length=512
        )

    tokenized_dataset = datasets.map(
        preprocess_function, batched=True, num_proc=10, remove_columns=["text"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=save_file,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        seed=42,
        dataloader_num_workers=10,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
    )

    #get latest file from directory
    def get_latest_file(dir):
        list_of_files = glob.glob(dir + '/*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    if train == True or classifier_model == None:
        config = AutoConfig.from_pretrained(mlm_model + "/config.json", num_labels=2, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(mlm_model, config=config, local_files_only=True)
        
        trainer = Trainer(model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["test"],
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics,
                        callbacks=[
                                EarlyStoppingCallback(early_stopping_patience=3)
                            ],  # evaluation dataset
                        )
        
        trainer.train()
        
        save_directory = get_latest_file(save_file)  

    else:
        config = AutoConfig.from_pretrained(classifier_model + "/config.json", num_labels=2, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(classifier_model, config=config, local_files_only=True)
        trainer = Trainer(model=model,
                    args=training_args,
                    eval_dataset=tokenized_dataset["test"],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics,)
       
    if predictions == True:
        predictions = trainer.predict(tokenized_dataset["test"])
        save_directory = get_latest_file(classifier_model)  
        return predictions, save_directory
    else:
        return save_directory