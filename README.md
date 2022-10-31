# Bias Detection and Mitigation using Model Cards

This library can be used to generate model cards highlighting issues and limitations affecting a machine learning model, in particular from a model bias point of view.

The library takes in input:
 - A pandas dataframe or a NumPy array containing the data used to train the model
 - A pandas dataframe or a NumPy array containing a batch of holdout data
 - The model predictions for the training and holdout data (and the corresponding probabilities when in presence of classification problems)
 - A list of  protected categories

The protected categories list of attributes that can be misrepresented or underrepresented in our dataset, such as gender, ethnicity, age, etc. 

Given  the inputs listed above,the library  creates a model cards containing a breakdown  of model performance metrics for different data slices describing protected categories.

## Data slices

A good approach to highlight data issues reflecting a machine learning model performance is to start reasoning in terms of data slices. This corresponds to analyzing each metric for subgroups of points defined by certain rules or characteristics. These subgroups are usually defined as data slices, i.e. data partitions obtained according to specific queries.

Performing such slicing and calculating metrics over each slice will give us a much more granular overview of our dataset's issues. 

Once we possess this information, we can augment our dataset by targeting specific data slices, making them less noisy or more represented within the whole dataset.

## Model card content

A model card is generated as a json file containing the following information:

- A global error matrix
- A list of data slices defining the model cards
- An error matrix for each data slice
- A model calibration curve
- A model bias summary
- A list of problematic data slices and a mitigation strategy

## Bias metrics 

The model bias summary is calculated with respect to the protected categories defined by the user.

The fairness metrics used within this library are:

- Equalised odds: which means measuring how much the model's predictions are conditionally dependent on the sensitive features. It corresponds to calculating TPR, and FPR across protected groups, measuring their imbalance.
- Equal opportunity: it corresponds to measuring how much the conditional expectations for positive labels change across protected groups.

The two metrics defined above are fairly easy to measure once we have a model and a set of protected categories. 

## Example

``` python
from bias_detection_mitigation.model_card import model_card

protected_categories = ['sex','age']
modelcard = model_card(X_train, X_test, y_train, y_test, model_pred_train, model_pred_test, protected_categories=protected_categories)



modelcard.save('card.txt')
```


## Funding
![img.png](img.png)