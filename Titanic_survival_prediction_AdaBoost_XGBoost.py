from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 1: Load the dataset and clean missing data
# Load the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df.dropna(subset=['Embarked'], inplace=True)  # Remove rows with missing 'Embarked'

# Drop irrelevant columns
df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Convert categorical features to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Step 2: Perform Exploratory Data Analysis (EDA)
# Visualize correlations between features
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Plot survival based on gender
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title('Survival based on Gender')
plt.show()

# Step 3: Model Training
# Split the data into training and testing sets
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train with Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train with Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
# Step 3 (extended): Train AdaBoost and XGBoost models

# AdaBoost Classifier
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)

# XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Step 4 (extended): Evaluate AdaBoost and XGBoost models

# Evaluate AdaBoost Model
ada_predictions = ada_model.predict(X_test)
print("AdaBoost Classifier Evaluation:")
print(confusion_matrix(y_test, ada_predictions))
print(classification_report(y_test, ada_predictions))

# Evaluate XGBoost Model
xgb_predictions = xgb_model.predict(X_test)
print("XGBoost Classifier Evaluation:")
print(confusion_matrix(y_test, xgb_predictions))
print(classification_report(y_test, xgb_predictions))

# Accuracy comparison for all models
print("AdaBoost Accuracy:", accuracy_score(y_test, ada_predictions))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_predictions))
