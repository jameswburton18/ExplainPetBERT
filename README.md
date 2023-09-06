## What do I need to do?
* [x] Check if anything is run on huggingface (Its not)
* [ ] Train the text model on all the columns
* [ ] Train the text model on the text column
* [ ] Move over the run shap code so i can apply the exact same thing to this new dataset

I think the plan should just be to repeat the same analysis that I did with the previous datasets with this one. No need to mess about with different combination methods. Although I think in Sean's paper he combined the predictions of the two modality models with a further decision tree/random forest. Either way that would be quite easy to implement actually. 

## Setup
```
pip install -e .
pip install -r requirements.txt
```

## Models
* [x] vet_40 F1 0.67757 fresh-plasma-14
* [x] vet_48 F1 0.64322 neat-cherry-33
* [x] vet_49 F1 0.67757 valiant-flower-30
* [x] vet_10 F1 0.6837 good-star-25
* [x] vet_18 F1 0.71102 fresh-capybara-27
* [x] vet_19 F1 0.72148 hardy-terrain-24