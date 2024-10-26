# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import LeakyReLU, Activation
from keras.optimizers import Adam
# Load the dataset
data = pd.read_csv('Titanic-Dataset.csv')

# Data Preprocessing
# Fill missing data
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical data to numeric using OneHotEncoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert target to categorical (for Keras compatibility)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



# Building an improved ANN model
model = Sequential()

# Input Layer & First Hidden Layer with Leaky ReLU
model.add(Dense(units=64, input_dim=X_train.shape[1]))  # Increase the units
model.add(LeakyReLU(alpha=0.1))  # Using Leaky ReLU activation
model.add(BatchNormalization())  # Batch Normalization for stability
model.add(Dropout(0.3))  # Dropout to prevent overfitting

# Second Hidden Layer with ELU
model.add(Dense(units=32))
model.add(Activation('elu'))  # ELU activation for faster convergence
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Third Hidden Layer with Swish
model.add(Dense(units=32))
model.add(Activation('swish'))  # Swish activation for potentially better performance
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(units=2, activation='softmax'))  # Using softmax for binary classification (2 outputs)

# Compile the model with a different optimizer
model.compile(optimizer=Adam(learning_rate=0.15), loss='binary_crossentropy', metrics=['accuracy'])

# Train the improved model
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'ANN Model Accuracy: {accuracy * 100:.2f}%')

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))
