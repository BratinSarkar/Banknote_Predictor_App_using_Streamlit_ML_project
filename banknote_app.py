# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 07:48:58 2021

@author: manis
"""
# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import numpy as np

# Step 2: Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
data = pd.read_csv(url, header=None, names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

# Step 3: Split Dataset
X = data[['variance', 'skewness', 'curtosis', 'entropy']]
y = data['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_report = classification_report(y_train, y_train_pred)
test_report = classification_report(y_test, y_test_pred)

train_confusion = confusion_matrix(y_train, y_train_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)

# Streamlit App Definition
def main():
    st.title("BankNote Authentication Classifier")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Model Performance
    st.subheader("Model Performance")
    st.write("Training Set Performance")
    st.write(f"Accuracy: {train_accuracy:.2f}")
    st.text("Classification Report")
    st.text(train_report)
    st.text("Confusion Matrix")
    st.text(np.array2string(train_confusion))

    st.write("Testing Set Performance")
    st.write(f"Accuracy: {test_accuracy:.2f}")
    st.text("Classification Report")
    st.text(test_report)
    st.text("Confusion Matrix")
    st.text(np.array2string(test_confusion))

    # User Input for Prediction
    st.subheader("Predict a BankNote")
    variance = st.number_input("Variance", value=0.0)
    skewness = st.number_input("Skewness", value=0.0)
    curtosis = st.number_input("Curtosis", value=0.0)
    entropy = st.number_input("Entropy", value=0.0)

    # Prediction Button
    if st.button("Predict"):
        prediction = model.predict([[variance, skewness, curtosis, entropy]])[0]
        st.write(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()
