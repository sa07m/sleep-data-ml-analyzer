# 🛏️ Sleep Data ML Analyzer

A Streamlit-based web app for visualizing sleep study data, detecting apnea-like events, and predicting sleep stages using machine learning.

This project is designed for students, healthcare data analysts, and job-seeking professionals looking to showcase real-world applications of data science in healthcare.

---

## 🚀 Features

✅ Upload and explore time-series sleep data  
✅ Visualize oxygen levels and heart rate over time  
✅ Automatically detect apnea-like events with customizable thresholds  
✅ Train a Random Forest model to classify sleep stages (Light, Deep, REM)  
✅ Display performance metrics and predictions interactively

---

## 🧠 Technologies Used

- Python (Pandas, NumPy)
- Scikit-learn (ML model)
- Plotly (interactive charts)
- Streamlit (web app)
- Git & GitHub (version control & deployment)

---

## 🗂️ Project Structure


sleep-data-ml-analyzer/
├── app/
│ └── app.py # Main Streamlit app
├── data/
│ └── sleep_data.csv # Sample sleep study dataset
├── utils/
│ └── ml_utils.py # ML model training function
├── requirements.txt # Python dependencies
└── README.md 

📊 Dataset Info
The sample sleep_data.csv file contains:

timestamp: minute-by-minute time series

oxygen_level: simulated oxygen saturation %

heart_rate: simulated heart rate in BPM

sleep_stage: labeled as Light, Deep, or REM

💡 Use Cases
Portfolio project for data science/healthcare analytics roles

Visual demo for interviews or academic presentations

Baseline for real-time bio-signal monitoring systems

🧪 Future Enhancements
Deep learning model for sleep stage detection

Integration with real medical data APIs

Export PDF sleep reports

User authentication for private datasets
