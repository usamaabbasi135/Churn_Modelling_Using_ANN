# Customer Churn Prediction using Artificial Neural Networks (ANN)

This project implements an end-to-end Artificial Neural Network (ANN) model to predict **customer churn** using TensorFlow and Keras. The model is trained on structured customer data from a bank, aiming to classify whether a customer is likely to leave (churn) or stay.

---

## Dataset

- **File:** `Churn_Modelling.csv`  
- **Source:** [Kaggle - Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)  
- **Records:** 10,000 customers  
- **Target Variable:** `Exited` (1 = churned, 0 = retained)

---

## ANN Architecture

| Layer Type     | Neurons | Activation |
|----------------|---------|------------|
| Input          | 11      | ReLU       |
| Hidden Layer 1 | 7       | ReLU       |
| Hidden Layer 2 | 6       | ReLU       |
| Output         | 1       | Sigmoid    |

- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Evaluation Metric:** Accuracy  
- **Callback:** EarlyStopping on `val_loss` with `patience=20`

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Workflow

### 1. Importing Libraries

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Loading Dataset

```python
dataset = pd.read_csv('/content/drive/MyDrive/Dataset/Churn_Modelling.csv')
```

### 3. Feature Selection

```python
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]
```

### 4. Feature Engineering

```python
geography = pd.get_dummies(X['Geography'], drop_first=True, dtype=int)
gender = pd.get_dummies(X['Gender'], drop_first=True, dtype=int)
X = X.drop(['Geography', 'Gender'], axis=1)
X = pd.concat([X, geography, gender], axis=1)
```

### 5. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### 6. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

---

## ANN Model Building

### Initialize the ANN

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

classifier = Sequential()
```

### Add Layers

```python
classifier.add(Dense(units=11, activation='relu'))
classifier.add(Dense(units=7, activation='relu'))
classifier.add(Dense(units=6, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))
```

### Compile the Model

```python
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=20,
    verbose=1,
    mode="auto",
    restore_best_weights=False
)
```

### Fit the Model

```python
model_history = classifier.fit(
    X_train, y_train,
    validation_split=0.33,
    batch_size=10,
    epochs=1000,
    callbacks=[early_stopping]
)
```

---

## Model Evaluation

### Plot Accuracy

```python
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Plot Loss

```python
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

---

## Predictions and Accuracy

```python
y_pred = classifier.predict(X_test)
y_pred = (y_pred >= 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy Score:", score)
```

---

## Getting Model Weights

```python
classifier.get_weights()
```

---

## Summary

- The ANN was able to reach around **86% validation accuracy**.
- It was trained with early stopping to avoid overfitting.
- Predictions were made using a 0.5 threshold for binary classification.

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

---

## Author

**Usama Abbasi**  
Data Analyst | ML | AI Engineer  
Islamabad, Pakistan

---

## Acknowledgments

- [Kaggle - Churn Modelling Dataset](https://www.kaggle.com/datasets/shubhendra7/customer-churn-prediction)
- TensorFlow / Keras Documentation
- Scikit-learn Documentation
