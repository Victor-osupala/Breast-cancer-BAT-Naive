import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from fpdf import FPDF
import base64
from io import BytesIO

# Load model and selected features
model = joblib.load("model_output/naive_bayes_model.pkl")
selected_features = joblib.load("model_output/selected_features.pkl")

all_feature_names = np.array([
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
])

selected_feature_names = all_feature_names[selected_features == 1]

# Feature defaults for sliders
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

# Helper: Save PDF
def generate_pdf(result, proba):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Breast Cancer Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Probability (Benign): {proba[0]:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"Probability (Malignant): {proba[1]:.4f}", ln=True)

    # Return PDF as string and encode to base64
    pdf_data = pdf.output(dest='S').encode('latin1')
    b64 = base64.b64encode(pdf_data).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'


# Streamlit UI
st.set_page_config(page_title="Breast Cancer App", layout="wide")
st.title("Breast Cancer Prediction System")

tabs = st.tabs(["🔍 Single Prediction", "📁 Upload & Batch Predict", "📊 SHAP Explanation"])

# TAB 1 — Single Prediction
with tabs[0]:
    st.header("Single Prediction")

    col1, col2 = st.columns([2, 1])
    user_input = []

    with st.sidebar:
      st.title("Input All Features")
      user_input = []
      for feature in all_feature_names:
          min_val, max_val, default_val = feature_defaults.get(feature, (0.0, 1.0, 0.5))
          val = st.slider(feature.replace("_", " ").title(), min_val, max_val, default_val)
          user_input.append(val)


      X_input = np.array(user_input).reshape(1, -1)
      X_input_selected = X_input[:, selected_features == 1]
      proba = model.predict_proba(X_input_selected)[0]
      prediction = model.predict(X_input_selected)[0]
      prediction_label = "Benign" if prediction == 0 else "Malignant"

    with col1:
        st.subheader("Feature Radar")
        plot_values = user_input + [user_input[0]]
        plot_labels = [f.replace("_", " ").title() for f in selected_feature_names] + [selected_feature_names[0].replace("_", " ").title()]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=plot_values, theta=plot_labels, fill='toself', name='Input', marker=dict(color='rgba(100,100,255,0.6)')))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        if prediction_label == "Benign":
            st.success("Benign")
        else:
            st.error("Malignant")

        st.markdown(f"**Probability of Benign:** `{proba[0]:.4f}`")
        st.markdown(f"**Probability of Malignant:** `{proba[1]:.4f}`")

        pdf_link = generate_pdf(prediction_label, proba)
        st.markdown(pdf_link, unsafe_allow_html=True)

# TAB 2 — Batch Upload
with tabs[1]:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not set(selected_feature_names).issubset(df.columns):
            st.warning("CSV must contain all selected features.")
        else:
            X = df[selected_feature_names]
            preds = model.predict(X)
            probs = model.predict_proba(X)

            df["Prediction"] = np.where(preds == 0, "Benign", "Malignant")
            df["Benign_Prob"] = probs[:, 0]
            df["Malignant_Prob"] = probs[:, 1]

            st.dataframe(df)

            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results as CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")

# # TAB 3 — SHAP
# with tabs[2]:
#     st.header("SHAP Explanation")

#     explainer = shap.Explainer(model.predict_proba, feature_names=selected_feature_names)
#     shap_values = explainer(X_input_selected)

#     st.set_option('deprecation.showPyplotGlobalUse', False)
#     st.subheader("SHAP Waterfall for Current Prediction")
#     shap.plots.waterfall(shap_values[0], max_display=10)
#     st.pyplot(bbox_inches='tight')
