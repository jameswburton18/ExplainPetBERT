#!/bin/bash
# for idx in 0 # {7..10}
# do
#     for cfg in airbnb_${idx}
#     do
#         sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
#         # sbatch --job-name=$cfg scripts/train.sh $cfg
#         sleep 4 # so that wandb runs don't get assigned the same number
#     done
# done

for idx in 48 49
do
    for cfg in vet_${idx}
    do
        sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
        # sbatch --job-name=$cfg scripts/train.sh $cfg
        sleep 4 # so that wandb runs don't get assigned the same number
    done
done