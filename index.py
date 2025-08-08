from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd
import os

# Load the model and columns just once when the function starts
model = joblib.load(os.path.join(os.path.dirname(__file__), 'alumni_match_model.joblib'))
model_columns = joblib.load(os.path.join(os.path.dirname(__file__), 'model_feature_columns.joblib'))

class handler(BaseHTTPRequestHandler):

    def _send_cors_headers(self):
        # This helper function sends the required permission headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        # This specifically handles the browser's "preflight" permission check
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        # Read the incoming data from the React app
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        incoming_data = json.loads(post_data)

        # --- PREPARE DATA FOR PREDICTION ---
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

        # --- MAKE PREDICTION ---
        prediction_proba = model.predict_proba(final_df)
        match_probability = prediction_proba[0][1]
        final_score = round(match_probability * 10)

        # --- SEND RESPONSE ---
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers() # Send permission headers with the actual response too
        self.end_headers()

        response = {'score': final_score}
        self.wfile.write(json.dumps(response).encode('utf-8'))
