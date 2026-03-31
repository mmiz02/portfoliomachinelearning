# Seminar 10 Preparation - Model Performance Measurement

The model_Performance_Measurement.ipnyb file was run and different parameters were changed to observe their impact on AUC and R2 error.

Firstly, the code mixed classification and regression metrics. Starting off with the confusion matrix (0,2,1,1) indicated poor model performance, with no correctly classified negative instances and a high number of false positives. This could suggest that the model is biased towards predicting the positive class and lacks reliability in distinguishing between classes.

F1 was calculated where results indicated that the model performs well for one clsss but fails completely for the other two classes. This leads to a low macro and weighted F1 score, showing lack of balance across classes. In the multilabel case, the model performed significantly better, achieving high scores, including a perfect score for one class. This indicates strong predictive performance and balanced classification across labels.

The ROC yielded an AUC of 0.79, indicating that the model had a reasonable ability to distinguish between classes. While performance was acceptable, it was not optimal, suggesting that further model improvements could be made. R2 was approximately 0.95 indicating that the model was a good fit overall.

### Changing Parameters

#### 1. Changing train / test split 
```python

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
 Havin an 80/20 training/testing split improved AUC with an area of 0.86. This is due to the model have more training data, allowing it to learn patterns more effectively.


 ```python
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
```
Having a 20/80 training/testing split reduced the AUC to 0.72. This demonstrates that not having enough training data will negatively impact the model's ability to generalise.

#### 2. Modify predictions (see how R2 changes)
```python
y_true = [3, -0.5, 2, 7]
y_pred = [3, -0.5, 2, 7]
```
The closer y_pred was to y_true, the higher R2 was. When they were the same R2 yielded 1 since it was a match. As they diverged, R2 decreased.

```python
y_true = [3, -0.5, 2, 7]
y_pred = [1, .3, 1, 3]
```
R2 was approximately 0.26 in the case above.

### Reflection

Overall, the results demonstrate that model performance is highly ensitive to parameter changes. Increasig the size of the training dataset improved classification performance (AUC), and poorer predictions significantly reduced regression performance (R2). This highlights the importance of proper data splitting and model tuning in achieving optimal predictive performance.

This task highlighted key ethical and professional considerations in machine learning. The variation in performance shows how model outcomes depend heavily on the parameters one chooses and the data quality. Poor results could lead to biased or unreliable decisions if deployed in practice.

This emphasises the responsibility of practitioners to ensure proper model validation, use of representative data, and transparency when working with data, taking decisions which are justifiable to the best of their knowledge.



