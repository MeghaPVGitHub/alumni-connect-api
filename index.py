from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd

model = joblib.load('api/alumni_match_model.joblib')
model_columns = joblib.load('api/model_feature_columns.joblib')

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        incoming_data = json.loads(post_data)

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

        final_df = df[model_columns]
        prediction_proba = model.predict_proba(final_df)
        match_probability = prediction_proba[0][1]
        final_score = round(match_probability * 10)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {'score': final_score}
        self.wfile.write(json.dumps(response).encode('utf-8'))
        return