import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Load dataset
def load_data():
    df = pd.read_csv("data/revenue_prediction.csv")
    df.drop(['Id', 'Name', 'Franchise', 'Category', 'City', 'No_Of_Item'], axis=1, inplace=True)
    return df

# Train and save model
def train_model(df):
    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        "iterations": [500, 1000, 1500],
        "learning_rate": [0.01, 0.03, 0.1],
        "depth": [4, 6, 8, 10],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128],
        "bagging_temperature": [0, 1, 2, 5],
    }
    
    model = CatBoostRegressor(verbose=0, random_state=42)
    random_search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    
    best_model = CatBoostRegressor(**random_search.best_params_, verbose=0, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Save model
    with open("model/model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    
    y_pred = best_model.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2 Score": r2_score(y_test, y_pred),
    }
    return metrics

# Load model
def load_model():
    with open("model/model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

# Streamlit UI
st.title("📊 Revenue Prediction App")
st.sidebar.header("Navigation")
option = st.sidebar.selectbox("Choose an option", ["Home", "Train Model", "Predict Revenue"])

df = load_data()

if option == "Home":
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.subheader("Data Distribution")
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Revenue"], bins=30, kde=True)
    st.pyplot(plt)
    
elif option == "Train Model":
    st.subheader("Model Training")
    if st.button("Train Model"):
        metrics = train_model(df)
        st.success("Model trained and saved successfully!")
        st.write(metrics)
    
elif option == "Predict Revenue":
    st.subheader("Make Predictions")
    model = load_model()
    order_placed = st.number_input("Enter Number of Orders Placed", min_value=1, step=1)
    if st.button("Predict"):
        prediction = model.predict(np.array([[order_placed]]))[0]
        st.success(f"Predicted Revenue: ${prediction:.2f}")
