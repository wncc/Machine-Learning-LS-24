# LOGISTICS REGRESSION 
***
Logistic regression is used for binary classification where we use a sigmoid function that takes input as independent variables and produces a probability value between 0 and 1.
To get an intuition refer to the [article](https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html).

For example, we have two classes Class 0 and Class 1 if the value of the logistic function for an input is greater than 0.5 (threshold value) then it belongs to Class 1 otherwise it belongs to Class 0. It's referred to as regression because it is the extension of linear regression but is mainly used for classification problems.This [Video(32 to 36)](https://www.youtube.com/watch?v=xuTiAW0OR40&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=32)
provides a deep insight to it.

>>Now , what happens at the backend 🤔. Let’s dive into its [maths](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc).



***
# REGULARISATION
***
>While developing machine learning models you must have encountered a situation in which the training accuracy of the model is high but the validation accuracy or the testing accuracy is low. This is the case which is popularly known as overfitting in the domain of machine learning. Also, this is the last thing a machine learning practitioner would like to have in his model. In this article, we will learn about a method known as Regularization in Python which helps us to solve the problem of overfitting. But before that let’s understand what is the role of regularisation  and what is underfitting and overfitting.
Refer to the [Video(37-39,41)](https://www.youtube.com/watch?v=8upNQi-40Q8&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=37) to get proper INTUITION.


Now, let's dive into types of logistic regression 
***
## BINARY  LOGISTIC REGRESSION
>In Binary Logistic regression, there can be only two possible types of the dependent variables, such as 0 or 1, Pass or Fail, etc. Refer to the [article](https://onezero.blog/modelling-binary-logistic-regression-using-python-research-oriented-modelling-and-interpretation/) to explore more. 

## MULTINOMIAL LOGISTIC REGRESSION
>In Multinomial Logistic Regression, there can be 3 or more possible unordered types of the dependent variable, such as "cat", "dogs", or "sheep”. Refer to the [article](https://www.pycodemates.com/2022/03/multinomial-logistic-regression-definition-math-and-implementation.html) to get better insights.
***
# FEATURE MAPPING
Feature mapping is a technique used in data analysis and machine learning to transform input data from a lower-dimensional space to a higher-dimensional space, where it can be more easily analyzed or classified
Go through the content below…
>One way to fit the data better is to create more features from each data point. We will map the features into all polynomial terms of
and
up to the sixth power. As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 28-dimensional vector. A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.
map_feature(x)=[x<sub>1</sub>,x<sub>2</sub>,x<sub>1</sub><sup>2</sup> ,x<sub>1</sub>x<sub>2</sub>,x<sub>1</sub><sup>3</sup>,....,x<sub>1</sub>x<sub>2</sub><sup>5</sup>,x<sub>2</sub><sup>6</sup>]
>
>This can be executed as
```python
 def map_feature(X1, X2):
     X1=np.atleast_1d(X1)
     X2=np.atleast_1d(X2)
     degree=6
     out=[]
     for i in range(1,degree+1):
       for j in range(i+1):
         out.append((X1**(i-j) * (X2**j)))
     return np.stack(out, axis=1)
```

This will be introduced in CNN in the further weeks.


# ASSIGNMENT
***
Plot decision boundary and find accuracy of your model on [data.txt](./data.txt)  
For further information refer the [colab_notebook](https://colab.research.google.com/drive/1oRnVWpXmK5JDKIHOOJm5bjwy8d9v6pRb#scrollTo=BqlxZOXoHh4z)
Save a copy of this in your drive . Compelete the blanks and save your colab notebook in github.
Share the github repo for verification.

