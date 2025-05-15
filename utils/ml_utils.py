
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_sleep_stage_model(df):
    df['sleep_stage_label'] = df['sleep_stage'].map({'Light': 0, 'Deep': 1, 'REM': 2})
    X = df[['oxygen_level', 'heart_rate']]
    y = df['sleep_stage_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    df['predicted_stage'] = model.predict(X)
    stage_map = {0: 'Light', 1: 'Deep', 2: 'REM'}
    df['predicted_stage'] = df['predicted_stage'].map(stage_map)
    return df, report
