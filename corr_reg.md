## Unit 3 Activity - Correlation and Regression - to check lecture cast

In this activity, four Jupyter notebooks were run to explore how changing datasets affect correlation and regression results. The notebooks were opened in the Jupyter Notebook interface and all cells were initially run to observe the baseline graphs and outputs.

The notebooks used were:
* covariance_pearson_correlation.ipynb;
* linear_regression.ipynb;
* multiple_linear_regression.ipynb;
* polynomial_regression.ipynb.

Data values were modified in the four notebooks to observe how the pattern of points affected the strength of correlation and regression.

### Ex 1 Covariance Pearson Correlation

Two datasets were generated to explore how correlation and covariance reflect the relationship between two variables. The original dataset *data2 = data1 + (10*randn(1000))* produced :
data1: mean=100.776 stdv=19.620 
data2: mean=151.050 stdv=22.358 
Covariance: 389.755 
Pearsons correlation: 0.888

The scatter plot showed a strong positive correlation.

Firstly, the data was tested by increasing noise:

'''python
data2 = data1 + (30*randn(1000))
'''
The Pearson correlation decreased to 0.549 and covariance reduced slightly due to more spread (399). The scatter became wider, showing weaker linear association.

Next, a negative relationship was introduced 

'''python
data2 = 200 - (data1 + (10*randn(1000)))
'''

The Pearson correlation became -0.883 with a negative covariance (-390), reflecting the inverse relationship. In addition, the scatter sloped downward.

A linear relation was then introduced:

'''python
data2 = data1 * 1.5
'''
The Pearson correlation = 1.0 with a covariance of 578 (vastly increased). The scatter plot formed a straight line, confirming perfect correlation. 

Finally, the random seed was changed (seed = 2):

With noise: Pearson correlation = 0.56, covariance changed slightly (397)
With negative relation: Pearson correlation = -0.901, covariance also negative (-401)
Linear relationship: Perfect correlation with an increased covariance (606)

This shows that both correlation and covariance depend on sample generation. The Pearson correlation reflects both the strength and direction of linear relationship. Covariance magnitude increases with joint variability, becoming negative for inverse relationships. Increasing noise reduces correlation and makes scatter more dispersed. Perfect linear relationships produce a correlation of 1 and a straight line scatter. Random seed generation affects the exact correlation and covariance values, demonstraing variability in random dstasets.

### Ex 2 Linear Regression

The relationship between two variables using Pearson correlation and linear regression was explored. The original dataset was:

'''python
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
'''

Pearson correlation = -0.759

The scatter plot showed a moderately strong negative correlation.

As an experiment, the first two y-values were modified to add outliers:
'''python
y[0] = 120
y[1] = 130
'''
This time, the Pearson correlation was -0.546 and the regression line slope decreased. Such outliers weakened the correlation and shifted the regression line, demonstrating that extreme data points can significantly influence both correlation and predictions.

For the next part of the exercise, the regression prediction at x = 10 was a speed of 85.59.
New y-values were predicted using:

'''python
def myfunc(x):
  return intercept + slope * x 

speed = myfunc(y)
'''

Where y = 15, 0 for which the outputs were 76.84 and 103.11 respectively.

This shows how predictions derease as x increases (negative slope) and confirms the the overall trend of the dataset is reflected in the regression.


### Ex 3 Multiple Linear Regression

Multiple linear regression was used to predict CO2 emissions based on Weight and Engine Volume from cars.csv.

Weight coefficient: 0.00755, meaning CO2 increases 0.00755 g per kg
Volume coefficient: 0.00781, meaning CO2 increases 0.00781 g per cm³

The baseline was as follows: Weight 2300 kg, Volume 1300 cm³, which led to CO2 emissions of 107.21 g.

Increasing the Weight by 1000 kg to 3300 kg and leaving the Volume at 1300 cm³, increased th CO2 emissions to 114.76 g.

This value was derived by taking the initial value of CO2 emissions and adding the multiplication of the Weight addition by the Weight coefficient:

107.2087328 + (1000 * 0.00755095) = 114.75968

This was confirmed by using the following code:
'''python
predictedCO2 = regr.predict([[3300, 1300]])
'''
The same output of 114.75968 was achieved.

As the weight or volume increased, so did the CO2 emissions. Every time the increase was proportional to the coefficient depending on whether the Weight, Volume, or both increased. Model predictions matched manual calculations and behaviour was as expected.


### Ex 4 Polynomial Regression
The following dataset was used to predict a value at x = 17:

'''python
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20] 
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
'''

This produced a prediction of approximately 90.55.

An extreme outlier was then added:

'''python
x.append(500) 
y.append(10)
'''

The prediction then changed to approximately 87.07.

This happened because polynomial regression fits one curve across all data points. The distant outlier distorted the curve shape and influenced predictions in areas far from the outlier.

In an earlier experiment using the following irregularly spaced large values:

'''python
x = [1,2,3,5,6,7,8,9,10,12,13,14,154,60,79,80,34,35] 
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
'''
The prediction was even lower (around 76.76), showing that widely spread datasets reduce prediction reliability.

### Conclusion

In this activity, correlation and regression were explored using multiple datasets. Through the four exercises, changes in data, outliers, and extreme values were shown to affect correlation coefficients, regression predictions, and the reliability of models.

Apart from the conclusions already derived above, one must also keep in mind ethical and professional considerations, where outliers or biased datasets can lead to misleading predictions. Machine learning professionals must carefully evaluate datasets to avoid producing results that could misinform decisions. Apart from this, experimenting with random and synthetic data highlighted variability and the influence of noise, reflecting challenges faced when dealing with real-life datasets.

[← Back to Home](https://mmiz02.github.io/portfoliomachinelearning/)
