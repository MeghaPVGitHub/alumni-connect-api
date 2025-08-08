from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

# This creates the main Flask application object that Vercel will use
app = Flask(__name__)

# This is the key: it tells the app to allow requests from any origin,
# which is what will fix the CORS error.
CORS(app)

# --- Load the model and columns ---
# This finds the files in the Vercel deployment environment
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'alumni_match_model.joblib')
columns_path = os.path.join(base_path, 'model_feature_columns.joblib')

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)


# Vercel looks for a single 'handler' or an 'app' object.
# By defining a route on the 'app' object, Vercel knows how to direct traffic.
@app.route('/', defaults={'path': ''}, methods=['POST', 'OPTIONS'])
@app.route('/<path:path>', methods=['POST', 'OPTIONS'])
def handler(path):
    # This handles the browser's pre-flight OPTIONS request
    if request.method == 'OPTIONS':
        # Create a response with the correct headers
        response = jsonify(success=True)
        return response

    # This handles the actual POST request with the data
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
