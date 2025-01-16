import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

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
    """
)

# Sidebar for user inputs
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

# Parameters for Isolation Forest
st.sidebar.title("Model Parameters")
contamination = st.sidebar.slider("Contamination (Anomaly Proportion)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
random_state = st.sidebar.number_input("Random State", value=42, step=1)

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
    selected_features = st.sidebar.multiselect("Select features for anomaly detection", options=numeric_columns, default=numeric_columns)

    if len(selected_features) > 0:
        # Preprocessing
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[selected_features])

        # Isolation Forest model
        model = IsolationForest(contamination=contamination, random_state=random_state)
        predictions = model.fit_predict(scaled_data)
        data["Anomaly"] = predictions
        data["Anomaly"] = data["Anomaly"].apply(lambda x: 1 if x == -1 else 0)

        # Display results
        st.write(f"Detected {data['Anomaly'].sum()} anomalies out of {len(data)} records.")

        # Visualize anomalies
        st.subheader("Anomaly Visualization")
        fig, ax = plt.subplots()
        ax.scatter(data.index, data[selected_features[0]], c=data["Anomaly"], cmap="coolwarm", label="Anomalies")
        ax.set_xlabel("Index")
        ax.set_ylabel(selected_features[0])
        ax.legend()
        st.pyplot(fig)

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
