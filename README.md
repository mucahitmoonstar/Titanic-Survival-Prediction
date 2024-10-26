Dataset Description:
"The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone on board, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

Acknowledgements:
This dataset has been referred from Kaggle: https://www.kaggle.com/c/titanic/data.

Objective:
Understand the Dataset & cleanup (if required).
Build a strong classification model to predict whether the passenger survives or not.
Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms."

GOAL:
"The main objective is to develop a predictive model that determines whether a passenger survived the Titanic disaster using various features, including age, gender, and class. The aim is to explore patterns and relationships that influenced survival odds. This model will not only predict outcomes accurately but also highlight the most critical factors impacting survival. By analyzing the Titanic dataset, this project demonstrates how data-driven methods can provide valuable insights into historical events and inform decision-making in similar future scenarios. The final model seeks to achieve high accuracy, offering reliable predictions based on the provided dataset."


ALGORİTHM:

Support Vector Machine (SVM)
Description: SVM finds a hyperplane that best separates the classes. It's effective for high-dimensional spaces and when there's a clear margin of separation.
Pros: Good with smaller datasets and non-linear boundaries.
Cons: Can be slow with large datasets and sensitive to feature scaling.

 K-Nearest Neighbors (KNN)
Description: KNN is a simple, instance-based learning algorithm that assigns a class to a sample based on the majority class of its nearest neighbors.
Pros: Simple to implement and interpret, no training phase.
Cons: Sensitive to feature scaling and the choice of K.

AdaBoost (Adaptive Boosting)
Description: AdaBoost is an ensemble method that combines multiple weak learners (usually decision trees) into a strong learner by focusing on hard-to-predict instances.
Pros: Improves accuracy by focusing on misclassified instances, reduces overfitting.
Cons: Sensitive to noisy data and outliers

XGBoost (Extreme Gradient Boosting)
Description: An advanced implementation of gradient boosting that uses tree boosting algorithms. It’s known for its speed and high performance.
Pros: High predictive power, handles missing data, strong regularization.
Cons: Complex to tune, higher memory consumption.

ANN  (Artificial Neural Networ)
An Artificial Neural Network (ANN) is a computational model inspired by the human brain. It consists of layers of interconnected nodes called neurons, which work together to recognize patterns in data. The network typically has three main types of layers:
Input Layer: Receives the raw data inputs (features).
Hidden Layers: Intermediate layers where neurons process data, learning complex patterns and relationships through weighted connections.
Output Layer: Produces the final prediction or classification result.
ANNs learn by adjusting the weights of connections using an optimization algorithm like backpropagation, which minimizes the error between predicted and actual outputs. They are especially effective for handling nonlinear relationships, complex data, and pattern recognition tasks, making them popular for applications like image recognition, language processing, and classification problems.

Random Forest;
Random Forest is an ensemble machine learning algorithm that consists of multiple decision trees. It combines the output of several trees to make a final prediction, which makes it more robust and accurate. Here's a breakdown:
How it Works: It creates a "forest" of decision trees, each trained on different random subsets of data and features. For classification, the final prediction is based on the majority vote (most common output) from all the trees. For regression, it averages the predictions.
Advantages: It reduces overfitting (compared to a single decision tree), handles missing values, and can manage large datasets with high dimensionality.
Applications: Random Forest is widely used for classification and regression tasks, such as predicting customer churn, disease diagnosis, and financial forecasting.


Logistic Regression:
Logistic Regression is a statistical model used for binary classification, predicting the probability of a binary outcome (0 or 1).
How it Works: It uses a linear equation to predict the probability of a class, applying the sigmoid function to constrain the output between 0 and 1. Based on a threshold (usually 0.5), the prediction is classified into one of the two categories.
Advantages: It's simple, easy to interpret, requires less computational power, and works well with linearly separable data.
Applications: Logistic Regression is used for binary classification tasks such as spam detection, credit scoring, and predicting medical conditions.

