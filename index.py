from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# Initialize the Flask app
app = Flask(__name__)
# This is the key change: it automatically handles CORS for all routes
CORS(app)

# --- Load the model and columns ---
# Vercel copies the files to a temporary directory, so we find the path
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'alumni_match_model.joblib')
columns_path = os.path.join(base_path, 'model_feature_columns.joblib')

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)


@app.route('/api', methods=['POST'])
def handle_request():
    # Get the JSON data sent from the React app
    incoming_data = request.get_json()

    # --- PREPARE THE DATA FOR PREDICTION ---
    df = pd.DataFrame([incoming_data])

    def count_common_skills(row):
        viewer_skills = set(str(row['viewer_skills']).lower().split('|'))
        target_skills = set(str(row['target_skills']).lower().split('|'))
        return len(viewer_skills.intersection(target_skills))

    df['common_skills_count'] = df.apply(count_common_skills, axis=1)
    df['branch_match'] = (df['viewer_branch'].str.lower() == df['target_branch'].str.lower()).astype(int)

    for col in model_columns:
        if col.startswith('company_'):
            df[col] = 0

    company_col_name = f"company_{incoming_data['target_company']}"
    if company_col_name in df.columns:
        df[company_col_name] = 1

    # Reorder columns to match the model's training order
    final_df = df[model_columns]

    # --- MAKE THE PREDICTION ---
    prediction_proba = model.predict_proba(final_df)
    match_probability = prediction_proba[0][1]
    final_score = round(match_probability * 10)

    # Send the score back as a JSON response
    return jsonify({'score': final_score})
