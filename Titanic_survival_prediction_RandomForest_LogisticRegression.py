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

# Step 4: Model Evaluation
# Evaluate Random Forest Model
rf_predictions = rf_model.predict(X_test)
print("Random Forest Classifier Evaluation:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# Evaluate Logistic Regression Model
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Evaluation:")
print(confusion_matrix(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# Accuracy comparison
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
