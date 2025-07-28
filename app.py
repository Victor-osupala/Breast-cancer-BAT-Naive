import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB

# Page configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

# Load model and selected features
model = joblib.load("breast_cancer_model_output/naive_bayes_model.pkl")
selected_features = joblib.load("breast_cancer_model_output/selected_features.pkl")

# Define all feature names
all_feature_names = np.array([
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
])

selected_feature_names = all_feature_names[selected_features == 1]

# Default slider values
feature_defaults = {
    "radius_mean": (0.0, 28.11, 14.13),
    "texture_mean": (0.0, 39.28, 19.29),
    "perimeter_mean": (0.0, 188.5, 91.97),
    "area_mean": (0.0, 2501.0, 654.89),
    "smoothness_mean": (0.0, 0.16, 0.10),
    "compactness_mean": (0.0, 0.35, 0.10),
    "concavity_mean": (0.0, 0.43, 0.09),
    "concave_points_mean": (0.0, 0.20, 0.05),
    "symmetry_mean": (0.0, 0.30, 0.15),
    "fractal_dimension_mean": (0.0, 0.10, 0.06),
    "radius_se": (0.0, 2.87, 0.5),
    "texture_se": (0.0, 4.88, 1.0),
    "perimeter_se": (0.0, 21.98, 8.0),
    "area_se": (0.0, 542.2, 40.0),
    "smoothness_se": (0.0, 0.03, 0.007),
    "compactness_se": (0.0, 0.14, 0.03),
    "concavity_se": (0.0, 0.4, 0.04),
    "concave_points_se": (0.0, 0.05, 0.01),
    "symmetry_se": (0.0, 0.08, 0.02),
    "fractal_dimension_se": (0.0, 0.03, 0.003),
    "radius_worst": (7.9, 36.04, 16.0),
    "texture_worst": (12.0, 49.54, 25.0),
    "perimeter_worst": (50.4, 251.2, 107.0),
    "area_worst": (185.2, 4254.0, 880.0),
    "smoothness_worst": (0.07, 0.22, 0.13),
    "compactness_worst": (0.0, 1.58, 0.25),
    "concavity_worst": (0.0, 1.3, 0.3),
    "concave_points_worst": (0.0, 0.29, 0.1),
    "symmetry_worst": (0.16, 0.66, 0.3),
    "fractal_dimension_worst": (0.05, 0.21, 0.08),
}

# Sidebar: sliders for features
st.sidebar.title("Nuclear Size Quantification")
user_input = []

for feature in selected_feature_names:
    if feature in feature_defaults:
        min_val, max_val, default_val = feature_defaults[feature]
    else:
        min_val, max_val, default_val = 0.0, 1.0, 0.5
    val = st.sidebar.slider(feature.replace("_", " ").title(), min_val, max_val, default_val)
    user_input.append(val)

X_input_selected = np.array(user_input).reshape(1, -1)

# Predict
proba = model.predict_proba(X_input_selected)[0]
prediction = model.predict(X_input_selected)[0]
prediction_label = "Benign" if prediction == 0 else "Malignant"

# Main header
st.title("Breast Cancer Predictor")
st.markdown(
    "This application combines cell nuclei measurement with machine learning techniques to predict "
    "breast cancer likelihood accurately. It provides healthcare professionals with advanced tools "
    "for precise cell nuclei analysis, contributing to improved breast cancer prediction and diagnosis."
)

# Two-column layout
col1, col2 = st.columns([2, 1])

# Column 1: Radar chart
with col1:
    st.subheader("Interactive Feature Radar Chart")

    plot_values = user_input + [user_input[0]]
    plot_labels = [f.replace("_", " ").title() for f in selected_feature_names] + [selected_feature_names[0].replace("_", " ").title()]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=plot_values,
        theta=plot_labels,
        fill='toself',
        name='Input',
        marker=dict(color='rgba(100, 100, 255, 0.6)')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=450, width=500
    )

    st.plotly_chart(fig, use_container_width=True)

# Column 2: Prediction info
with col2:
    st.subheader("Cell Cluster Prediction")
    st.markdown("#### The cell cluster is:")
    if prediction_label == "Benign":
        st.success("Benign")
    else:
        st.error("Malignant")

    st.markdown(f"**Probability of being Benign:** `{proba[0]:.6f}`")
    st.markdown(f"**Probability of being Malicious:** `{proba[1]:.6f}`")

    st.info(
        "This app can assist medical professionals in making a diagnosis, "
        "but should not be used as a substitute for a professional diagnosis."
    )
