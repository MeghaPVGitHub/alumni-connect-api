from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import requests
import io

app = Flask(__name__)
CORS(app)

# --- Load Models from Hugging Face ---
# PASTE THE URLS YOU COPIED FROM HUGGING FACE HERE
MODEL_URL = "https://huggingface.co/Megzz22/alumni-connect-model/resolve/main/alumni_match_model.joblib"
COLUMNS_URL = "https://huggingface.co/Megzz22/alumni-connect-model/resolve/main/model_feature_columns.joblib"

# Download and load the model
model_res = requests.get(MODEL_URL)
model = joblib.load(io.BytesIO(model_res.content))

# Download and load the column list
columns_res = requests.get(COLUMNS_URL)
model_columns = joblib.load(io.BytesIO(columns_res.content))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def handler(path):
    if request.method == 'OPTIONS':
        return jsonify(success=True)

    if request.method == 'POST':
        incoming_data = request.get_json()
        df = pd.DataFrame([incoming_data])

        def count_common_skills(row):
            viewer_skills = set(str(row.get('viewer_skills', '')).lower().split('|'))
            target_skills = set(str(row.get('target_skills', '')).lower().split('|'))
            return len(viewer_skills.intersection(target_skills))

        df['common_skills_count'] = df.apply(count_common_skills, axis=1)
        df['branch_match'] = (df['viewer_branch'].str.lower() == df['target_branch'].str.lower()).astype(int)

        for col in model_columns:
            if col.startswith('company_'):
                df[col] = 0

        company_name = incoming_data.get('target_company', '')
        if company_name:
            company_col_name = f"company_{company_name}"
            if company_col_name in df.columns:
                df[company_col_name] = 1

        final_df = df[model_columns]
        prediction_proba = model.predict_proba(final_df)
        match_probability = prediction_proba[0][1]
        final_score = round(match_probability * 10)

        return jsonify({'score': final_score})
