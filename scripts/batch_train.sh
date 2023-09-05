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

for cfg in all_text #airbnb_${idx} salary_${idx} channel_${idx} fake_${idx} kick_${idx} jigsaw_${idx} wine_${idx}  imdb_genre_${idx}  
do
    sbatch --job-name=$cfg scripts/train_office_pc.sh $cfg
    # sbatch --job-name=$cfg scripts/train.sh $cfg
    sleep 4 # so that wandb runs don't get assigned the same number
done