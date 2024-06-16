#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import ttk, messagebox


# In[9]:


# Load the dataset
data = pd.read_csv('clean_dataset.csv')

# Display basic statistics of the dataset
print(data.describe())

# Separate features (X) and target (y)
X = data.drop('Approved', axis=1)
y = data['Approved']

# Define categorical and numerical features
categorical_features = ['Gender', 'Married', 'BankCustomer', 'Industry', 'Ethnicity', 'PriorDefault', 'Employed', 'DriversLicense', 'Citizen']
numerical_features = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'ZipCode', 'Income']

# Preprocessing pipeline for numerical features
numerical_transformer = StandardScaler()

# Preprocessing pipeline for categorical features
categorical_transformer = OneHotEncoder(drop='first')

# Combine preprocessing pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)


# In[10]:


# Create the main application window
root = tk.Tk()
root.title("Loan Approval Predictor")

# Function to make predictions and display results
def predict_and_display_plots():
    try:
        # Get user input from GUI widgets
        new_applicant_data = {
            'Gender': int(gender_combobox.get()), 
            'Age': int(age_entry.get()),
            'Debt': int(debt_entry.get()), 
            'Married': int(married_combobox.get()),
            'BankCustomer': int(bank_customer_combobox.get()), 
            'Industry': industry_entry.get(),
            'Ethnicity': ethnicity_entry.get(), 
            'YearsEmployed': float(years_employed_entry.get()), 
            'PriorDefault': int(prior_default_combobox.get()), 
            'Employed': int(employed_combobox.get()), 
            'CreditScore': int(credit_score_entry.get()),
            'DriversLicense': int(drivers_license_combobox.get()), 
            'Citizen': citizen_entry.get(),
            'ZipCode': int(zip_code_entry.get()), 
            'Income': int(income_entry.get())
        }

        # Reshape the input data into DataFrame with a single row
        new_applicant_df = pd.DataFrame([new_applicant_data])

        # Make a prediction using the trained model
        prediction = pipeline.predict(new_applicant_df)
        if prediction[0] == 1:
            prediction_result = "Approved"
            result_label.config(text="Prediction: Approved", foreground="green")
            reason = "All thresholds met"
        else:
            prediction_result = "Rejected"
            result_label.config(text="Prediction: Rejected", foreground="red")

            # Get the reason for rejection
            reason = reason_for_approval_rejection(new_applicant_data)

        # Call display_plots with the new_applicant_data as an argument
        display_plots(new_applicant_data)
        plot_threshold_comparison(new_applicant_data)

        # Display the result and reason in a message box
        messagebox.showinfo("Prediction Result", f"Prediction: {prediction_result}\nReason: {reason}")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numerical values.")

# Function to adjust thresholds based on clusters
def adjust_thresholds_based_on_clusters(X_cluster, kmeans, new_applicant_data, debt_threshold, credit_score_threshold, income_threshold):
    # Calculate centroids for each cluster
    centroids = kmeans.cluster_centers_

    # Predict cluster labels
    cluster_labels = kmeans.predict(X_cluster)

    # Find majority class within each cluster
    cluster_majority_class = []
    for label in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == label)[0]
        cluster_approved_count = np.sum(y_test.iloc[cluster_indices] == 1)
        cluster_not_approved_count = len(cluster_indices) - cluster_approved_count
        majority_class = 1 if cluster_approved_count > cluster_not_approved_count else 0
        cluster_majority_class.append(majority_class)

    # Calculate Euclidean distance between centroids and new applicant data
    distances = []
    for centroid in centroids:
        try:
            distance = np.linalg.norm(centroid - new_applicant_data.values)
            distances.append(distance)
        except TypeError as e:
            print("Error:", e)
            print("Centroid:", centroid)
            print("New Applicant Data:", new_applicant_data.values)

    # Adjust thresholds based on distances and majority class of clusters
    for idx, distance in enumerate(distances):
        if cluster_majority_class[idx] == 1:
            if distance < 0.1:  # Adjust this threshold based on the scale of your data
                # Adjust thresholds for approved class
                debt_threshold = np.percentile(data['Debt'], 75)
                credit_score_threshold = np.percentile(data['CreditScore'], 25)
                income_threshold = np.percentile(data['Income'], 75)
        else:
            if distance < 0.1:  # Adjust this threshold based on the scale of your data
                # Adjust thresholds for not approved class
                debt_threshold = np.percentile(data['Debt'], 75)
                credit_score_threshold = np.percentile(data['CreditScore'], 25)
                income_threshold = np.percentile(data['Income'], 75)

    return debt_threshold, credit_score_threshold, income_threshold

# Function to display plots
def display_plots(new_applicant_data):
    plot_knn(new_applicant_data)

    # Concatenate numerical and one-hot encoded features for X_cluster
    global X_cluster, kmeans
    X_cluster = pd.concat([X_test[numerical_features], pd.get_dummies(X_test[categorical_features], drop_first=True)], axis=1)

    # Call plot_kmeans_clusters with appropriate arguments
    plot_kmeans_clusters(X_cluster, new_applicant_data)
    plot_approved_not_approved_separately(new_applicant_data, X_test, y_test)

# Function to plot KNN
def plot_knn(new_applicant_data):
    # Preprocessing pipeline for categorical features
    categorical_transformer_knn = OneHotEncoder(drop='first')

    # Combine preprocessing pipelines using ColumnTransformer for KNN
    preprocessor_knn = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer_knn, categorical_features)
        ])

    # Define the model pipeline for KNN
    pipeline_knn = Pipeline([
        ('preprocessor', preprocessor_knn),
        ('classifier', KNeighborsClassifier())
    ])

    # Split data into train and test sets
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the KNN model
    pipeline_knn.fit(X_train_knn, y_train_knn)

    # Make predictions using KNN
    y_pred_knn = pipeline_knn.predict(X_test_knn)

    # Evaluate KNN model
    accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
    print(f"KNN Accuracy: {accuracy_knn}")

    # Plot KNN results (Add your plotting code here)

# Function to plot KMeans clusters
def plot_kmeans_clusters(X_cluster, new_applicant_data):
    # KMeans Clustering
    global kmeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)  # Explicitly set n_init
    kmeans.fit(X_cluster)

    # Predict cluster labels
    cluster_labels = kmeans.predict(X_cluster)

    # Visualize clusters and classification
    plt.figure(figsize=(12, 6))

    # Plot clusters based on income and credit score
    plt.subplot(1, 2, 1)
    plt.scatter(X_cluster['Income'], X_cluster['CreditScore'], c=cluster_labels, cmap='viridis', alpha=0.5)
    plt.scatter(new_applicant_data['Income'], new_applicant_data['CreditScore'], marker='*', s=300, c='blue', label='New Applicant')
    plt.xlabel('Income')
    plt.ylabel('Credit Score')
    plt.title('Clusters based on Income and Credit Score')
    plt.legend()

    # Plot classification based on KNN (Add your plotting code here)

    plt.tight_layout()
    plt.show()

# Function to plot approved vs. not approved applicants
def plot_approved_not_approved_separately(new_applicant_data, X_test, y_test):
    # Extract approved and not approved data from the test set
    approved_data = X_test[y_test == 1]
    not_approved_data = X_test[y_test == 0]

    # Plot Approved vs. Not Approved
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(approved_data['Income'], approved_data['CreditScore'], c='g', alpha=0.5, label='Approved')
    plt.scatter(not_approved_data['Income'], not_approved_data['CreditScore'], c='r', alpha=0.5, label='Not Approved')
    plt.scatter(new_applicant_data['Income'], new_applicant_data['CreditScore'], marker='*', s=300, c='blue', label='New Applicant')
    plt.xlabel('Income')
    plt.ylabel('Credit Score')
    plt.title('Approved vs. Not Approved Applicants')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(approved_data['Debt'], approved_data['YearsEmployed'], c='g', alpha=0.5, label='Approved')
    plt.scatter(not_approved_data['Debt'], not_approved_data['YearsEmployed'], c='r', alpha=0.5, label='Not Approved')
    plt.scatter(new_applicant_data['Debt'], new_applicant_data['YearsEmployed'], marker='*', s=300, c='blue', label='New Applicant')
    plt.xlabel('Debt')
    plt.ylabel('Years Employed')
    plt.title('Approved vs. Not Approved Applicants')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to plot threshold comparison
def plot_threshold_comparison(new_applicant_data):
    # Define thresholds for comparison (these can be adjusted as needed)
    debt_threshold = 280  # Example threshold, replace with your logic
    credit_score_threshold = 5  # Example threshold, replace with your logic
    income_threshold = 1017  # Example threshold, replace with your logic

    # Adjust thresholds based on clusters
    debt_threshold, credit_score_threshold, income_threshold = adjust_thresholds_based_on_clusters(
        X_cluster, kmeans, new_applicant_data, debt_threshold, credit_score_threshold, income_threshold)

    # Plot comparison
    plt.figure(figsize=(10, 6))

    plt.bar(['Debt', 'Credit Score', 'Income'], [new_applicant_data['Debt'], new_applicant_data['CreditScore'], new_applicant_data['Income']], label='Applicant', alpha=0.6)
    plt.axhline(debt_threshold, color='r', linestyle='--', label='Debt Threshold')
    plt.axhline(credit_score_threshold, color='g', linestyle='--', label='Credit Score Threshold')
    plt.axhline(income_threshold, color='b', linestyle='--', label='Income Threshold')

    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title('Threshold Comparison')
    plt.legend()

    plt.show()

# Function to get the reason for approval/rejection
def reason_for_approval_rejection(new_applicant_data):
    # Define your thresholds for approval/rejection
    debt_threshold = 280  # Example threshold, replace with your logic
    credit_score_threshold = 5  # Example threshold, replace with your logic
    income_threshold = 1017  # Example threshold, replace with your logic

    # Adjust thresholds based on clusters
    debt_threshold, credit_score_threshold, income_threshold = adjust_thresholds_based_on_clusters(
        X_cluster, kmeans, new_applicant_data, debt_threshold, credit_score_threshold, income_threshold)

    reasons = []

    if new_applicant_data['Debt'] > debt_threshold:
        reasons.append('Debt exceeds threshold')
    if new_applicant_data['CreditScore'] < credit_score_threshold:
        reasons.append('Credit score below threshold')
    if new_applicant_data['Income'] < income_threshold:
        reasons.append('Income below threshold')

    if not reasons:
        reasons.append('All thresholds met')
    return ', '.join(reasons)

# Define the GUI layout and widgets
main_frame = ttk.Frame(root, padding="10")
main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Create labels and input widgets for each feature
ttk.Label(main_frame, text="Gender (0=Female, 1=Male):").grid(row=0, column=0, sticky=tk.W)
gender_combobox = ttk.Combobox(main_frame, values=[0, 1])
gender_combobox.grid(row=0, column=1)

ttk.Label(main_frame, text="Age:").grid(row=1, column=0, sticky=tk.W)
age_entry = ttk.Entry(main_frame)
age_entry.grid(row=1, column=1)

ttk.Label(main_frame, text="Debt:").grid(row=2, column=0, sticky=tk.W)
debt_entry = ttk.Entry(main_frame)
debt_entry.grid(row=2, column=1)

ttk.Label(main_frame, text="Married (0=No, 1=Yes):").grid(row=3, column=0, sticky=tk.W)
married_combobox = ttk.Combobox(main_frame, values=[0, 1])
married_combobox.grid(row=3, column=1)

ttk.Label(main_frame, text="BankCustomer (0=No, 1=Yes):").grid(row=4, column=0, sticky=tk.W)
bank_customer_combobox = ttk.Combobox(main_frame, values=[0, 1])
bank_customer_combobox.grid(row=4, column=1)

ttk.Label(main_frame, text="Industry:").grid(row=5, column=0, sticky=tk.W)
industry_entry = ttk.Entry(main_frame)
industry_entry.grid(row=5, column=1)

ttk.Label(main_frame, text="Ethnicity:").grid(row=6, column=0, sticky=tk.W)
ethnicity_entry = ttk.Entry(main_frame)
ethnicity_entry.grid(row=6, column=1)

ttk.Label(main_frame, text="Years Employed:").grid(row=7, column=0, sticky=tk.W)
years_employed_entry = ttk.Entry(main_frame)
years_employed_entry.grid(row=7, column=1)

ttk.Label(main_frame, text="Prior Default (0=No, 1=Yes):").grid(row=8, column=0, sticky=tk.W)
prior_default_combobox = ttk.Combobox(main_frame, values=[0, 1])
prior_default_combobox.grid(row=8, column=1)

ttk.Label(main_frame, text="Employed (0=No, 1=Yes):").grid(row=9, column=0, sticky=tk.W)
employed_combobox = ttk.Combobox(main_frame, values=[0, 1])
employed_combobox.grid(row=9, column=1)

ttk.Label(main_frame, text="Credit Score:").grid(row=10, column=0, sticky=tk.W)
credit_score_entry = ttk.Entry(main_frame)
credit_score_entry.grid(row=10, column=1)

ttk.Label(main_frame, text="Driver's License (0=No, 1=Yes):").grid(row=11, column=0, sticky=tk.W)
drivers_license_combobox = ttk.Combobox(main_frame, values=[0, 1])
drivers_license_combobox.grid(row=11, column=1)

ttk.Label(main_frame, text="Citizen:").grid(row=12, column=0, sticky=tk.W)
citizen_entry = ttk.Entry(main_frame)
citizen_entry.grid(row=12, column=1)

ttk.Label(main_frame, text="Zip Code:").grid(row=13, column=0, sticky=tk.W)
zip_code_entry = ttk.Entry(main_frame)
zip_code_entry.grid(row=13, column=1)

ttk.Label(main_frame, text="Income:").grid(row=14, column=0, sticky=tk.W)
income_entry = ttk.Entry(main_frame)
income_entry.grid(row=14, column=1)

# Button to make a prediction and display the result
predict_button = ttk.Button(main_frame, text="Predict", command=predict_and_display_plots)
predict_button.grid(row=15, column=0, columnspan=2, pady=10)

# Label to display the result
result_label = ttk.Label(main_frame, text="Prediction: ", font=("Helvetica", 12))
result_label.grid(row=16, column=0, columnspan=2)

# Run the Tkinter event loop
root.mainloop()


# In[ ]:




