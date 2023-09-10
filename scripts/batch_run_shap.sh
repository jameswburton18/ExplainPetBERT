#!/bin/bash
for idx in 1 #50 #10a 18a 19a 40a 48a 
do
    for cfg in vet_10b_baseline vet_19b_ensemble_50 vet_19b_ensemble_25 vet_19b_ensemble_75 #vet_10b_all_text
    do
        # sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        sbatch --job-name=$cfg scripts/run_shap.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done