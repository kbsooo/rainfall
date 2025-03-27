# [Binary Prediction with a Rainfall Dataset](https://www.kaggle.com/competitions/playground-series-s5e3)
Playground Series - Season 5, Episode 3

![](https://www.kaggle.com/competitions/91714/images/header)

### Overview
Welcome to the 2025 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting and approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: Your goal is to predict rainfall for each day of the year.

### Evaluation
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

### Submission File
For each `id` in the test set, you must predict a probability for the target `rainfall`. The file should contain a header and have the following format:

```
id,rainfall
2190,0.5
2191,0.1
2192,0.9
etc.
```

### Citation
Walter Reade and Elizabeth Park. Binary Prediction with a Rainfall Dataset. https://kaggle.com/competitions/playground-series-s5e3, 2025. Kaggle.