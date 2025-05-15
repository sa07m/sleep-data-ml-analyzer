
import sys
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Fix import path so Streamlit can locate 'utils'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.ml_utils import train_sleep_stage_model

st.set_page_config(page_title="Sleep Pattern Analyzer with ML", layout="wide")

st.title("Sleep Pattern Analyzer with ML Sleep Stage Prediction")

uploaded_file = st.file_uploader("Upload Sleep Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Sleep Data")
    st.dataframe(df.head())

    time_col = "timestamp"
    metric_col = st.selectbox("Select Metric to Visualize", ['oxygen_level', 'heart_rate'])

    st.subheader("Time-Series Plot")
    fig = px.line(df, x=time_col, y=metric_col, title=f"{metric_col.replace('_',' ').title()} over Time")
    st.plotly_chart(fig, use_container_width=True)

    threshold = st.slider("Set Apnea Detection Threshold", float(df[metric_col].min()), float(df[metric_col].max()), float(df[metric_col].mean()))
    events = df[df[metric_col] < threshold]
    st.subheader("Detected Apnea Events")
    st.write(f"Detected {len(events)} events below threshold")
    st.dataframe(events[['timestamp', metric_col]])

    st.subheader("ML Model: Predict Sleep Stages")
    if 'sleep_stage' in df.columns:
        df, report = train_sleep_stage_model(df)
        st.text("Classification Report:")
        st.dataframe(pd.DataFrame(report).transpose())
        st.subheader("Full Dataset with Predicted Sleep Stages")
        st.dataframe(df[['timestamp', 'oxygen_level', 'heart_rate', 'predicted_stage']])
    else:
        st.warning("Sleep stage column not found in the dataset.")
