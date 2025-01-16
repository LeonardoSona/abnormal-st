import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap

# Streamlit app title
st.title("Abnormal Usage Detection Tool")

# Key notes for users
st.subheader("Getting Started")
st.markdown(
    """
    **Key Notes:**
    - Upload a CSV file containing numerical data for anomaly detection.
    - Ensure the dataset includes rows as records (e.g., user sessions) and columns as features (e.g., UsageDuration, Clicks).
    - Avoid missing or invalid values in the dataset.
    - The app scales the data automatically, so different feature ranges are handled appropriately.
    - Example columns:
        - `UsageDuration`: Duration of usage in minutes.
        - `Clicks`: Number of clicks performed in a session.
        - `Transactions`: Number of transactions completed.
        - `SessionTime`: Time spent in the system.
    - **Contamination:** Proportion of data expected to be anomalies (default is 10%). Adjust based on your use case.
    - **Random State:** Ensures reproducibility by fixing the random seed for the algorithm.
    - You can concatenate columns (e.g., UserID and AppID) to create unique identifiers for anomaly detection.
    - **Interpretability:** Anomaly detection focuses on identifying outliers, but explanations can be provided using SHAP for feature contributions.
    """
)

# Sidebar for user inputs
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Parameters for Isolation Forest
st.sidebar.title("Model Parameters")
contamination = st.sidebar.slider("Contamination (Anomaly Proportion)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Proportion of the dataset expected to be anomalies.")
random_state = st.sidebar.number_input("Random State", value=42, step=1, help="Fixes the random seed for reproducibility.")

# Placeholder for results
st.subheader("Detection Results")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(data.head())

    # Feature selection
    st.sidebar.title("Feature Selection")
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Select features for anomaly detection", 
        options=numeric_columns + non_numeric_columns, 
        default=numeric_columns
    )

    # Option to concatenate columns for unique identifiers
    st.sidebar.title("Advanced Options")
    concat_columns = st.sidebar.multiselect(
        "Select columns to concatenate for unique identifiers", 
        options=non_numeric_columns
    )

    if len(concat_columns) > 0:
        data["Concatenated"] = data[concat_columns].astype(str).agg("_".join, axis=1)
        selected_features.append("Concatenated")

    if len(selected_features) > 0:
        # Preprocessing
        for col in non_numeric_columns:
            if col in selected_features and col != "Concatenated":
                data[col] = data[col].astype("category").cat.codes

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[selected_features])

        # Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=random_state)
        predictions = model.fit_predict(scaled_data)
        data["Anomaly"] = predictions
        data["Anomaly"] = data["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

        # Display results
        st.write(f"Detected {data['Anomaly'].sum()} anomalies out of {len(data)} records.")

        # List of anomalies for verification
        st.subheader("List of Anomalous Records")
        anomalies = data[data["Anomaly"] == 1]
        st.dataframe(anomalies)

        # Visualize anomalies
        st.subheader("Anomaly Visualization")
        fig, ax = plt.subplots()
        ax.scatter(data.index, data[selected_features[0]], c=data["Anomaly"], cmap="coolwarm", label="Anomalies")
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_features[0])
        ax.legend()
        st.pyplot(fig)
        
        # Interpretability using SHAP (KernelExplainer)
        st.subheader("Model Interpretability with SHAP")
        compute_shap = st.checkbox("Compute SHAP values (may take time for large datasets)", value=False)

        if compute_shap:
            # Using KernelExplainer as IsolationForest is not natively supported
            background_sample = scaled_data[np.random.choice(scaled_data.shape[0], size=min(100, len(scaled_data)), replace=False)]
            explainer = shap.KernelExplainer(model.predict, background_sample)
            shap_values = explainer.shap_values(scaled_data, nsamples=100)

        st.write("SHAP Summary Plot:")
        shap.summary_plot(shap_values, pd.DataFrame(scaled_data, columns=selected_features))
        st.pyplot()
        else:
            st.write("Enable SHAP computation to interpret feature contributions.")
        
        # Feature importance explanation (approximation)
        st.subheader("Feature Importance")
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": selected_features, "Importance": importance})
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            st.bar_chart(importance_df.set_index("Feature"))
        else:
            st.write("Feature importance is not available for the Isolation Forest model.")
    else:
        st.write("Please select at least one feature for anomaly detection.")
else:
    st.write("Please upload a CSV file to start.")
