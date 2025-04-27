import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.stem import WordNetLemmatizer
import os

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize Flask app
app = Flask(__name__)

# Function to load CSV data
def load_data(file_path):
    try:
        data = pd.read_csv(file_path, encoding="latin1")
        return data
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return pd.DataFrame()

# Function to flatten and preprocess data
def flatten_and_preprocess(data):
    data = data.to_numpy()
    data = [item for item in data.ravel() if str(item) != "nan"]
    data = ["_".join(str(i).split()) for i in data]
    return data

# Function to recommend meals based on health conditions
def recommend_meals(conditions):
    datasets = {
        "diabetes": r"input/diabetes.csv",
        "heart_disease": r"input/heart_disease.csv",
        "high_blood_pressure": r"A:\project_x\AI-powered Nutrition Advisor\input\high_blood_pressure.csv",
        "liver_disease": r"A:\project_x\AI-powered Nutrition Advisor\input\Liver_disease.csv",
        "stroke": r"A:\project_x\AI-powered Nutrition Advisor\input\stroke.csv",
        "kidney_disease": r"A:\project_x\AI-powered Nutrition Advisor\input\kidney_disease.csv"
    }

    combined_data = []
    for condition in conditions:
        if condition in datasets:
            data = load_data(datasets[condition])
            combined_data.extend(flatten_and_preprocess(data))

    if not combined_data:
        return []

    lemmatizer = WordNetLemmatizer()
    combined_data = [lemmatizer.lemmatize(item) for item in combined_data]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(combined_data)
    feature_names = vectorizer.get_feature_names()

    model = KMeans(n_clusters=3, init="k-means++", max_iter=50, n_init=10, random_state=42)
    model.fit(X)

    safe_food = model.cluster_centers_.argsort()[0, -1:-26:-1]
    recommended_meals = [feature_names[word] for word in safe_food]

    return recommended_meals

# Sample precautions mapping based on severity
precautions_mapping = {
    "diabetes": {
        "low": "Maintain a balanced diet, exercise daily, and monitor blood sugar levels.",
        "medium": "Avoid sugary foods, monitor carb intake, and consult a dietitian.",
        "high": "Strict dietary restrictions, insulin therapy, and regular doctor visits."
    },
    "heart_disease": {
        "low": "Reduce salt intake, eat heart-healthy fats, and stay active.",
        "medium": "Limit saturated fats, monitor blood pressure, and moderate activity.",
        "high": "Strict low-sodium diet, medication, and supervised exercise."
    },
    "high_blood_pressure": {
        "low": "Reduce sodium intake, increase potassium-rich foods, and exercise regularly.",
        "medium": "Monitor blood pressure closely, limit processed foods, and manage stress.",
        "high": "Strict low-sodium diet, prescribed medication, and regular medical check-ups."
    },
    "liver_disease": {
        "low": "Avoid alcohol, eat a balanced diet, and stay hydrated.",
        "medium": "Limit high-fat foods, avoid toxins, and increase antioxidant-rich foods.",
        "high": "Follow a liver-friendly diet, take prescribed medication, and avoid all alcohol."
    },
    "stroke": {
        "low": "Maintain a balanced diet, stay physically active, and monitor blood pressure.",
        "medium": "Avoid high-sodium foods, monitor cholesterol levels, and practice stress management.",
        "high": "Strict dietary modifications, take prescribed medications, and regular therapy."
    },
    "kidney_disease": {
        "low": "Stay hydrated, limit salt intake, and eat kidney-friendly foods.",
        "medium": "Monitor protein intake, avoid high-potassium foods, and limit phosphorus.",
        "high": "Follow a strict renal diet, avoid processed foods, and undergo regular dialysis."
    }
}

# Flask routes
@app.route('/')
def home():
    return render_template('page1.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    conditions = request.form.getlist('conditions')
    severity = request.form.get('severity', 'low')

    recommendations = recommend_meals(conditions)
    print("Rec: ",recommendations)

    precautions = {
        condition: precautions_mapping.get(condition, {}).get(severity, "No precautions available.")
        for condition in conditions
    }

    return render_template('page2.html', recommendations=recommendations, precautions=precautions, selected_conditions=conditions, selected_severity=severity)

if __name__ == "__main__":
    app.run(debug=True)
