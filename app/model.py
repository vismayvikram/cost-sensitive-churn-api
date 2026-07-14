import xgboost as xgb
from xgboost import XGBClassifier
import shap
import json
import numpy as np
import pandas as pd

class ChurnModel:
    def __init__(self, model_path='churn_model_final.json', config_path='model_config.json', features_path='feature_names.json'):
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        with open(config_path) as f:
            config = json.load(f)
        self.threshold = config['optimal_threshold']
        self.fn_cost_multiplier = config['fn_cost_multiplier']
        self.fp_cost = config['fp_cost']

        with open(features_path) as f:
            self.model_feature_names = json.load(f)

        # Build the SHAP explainer once at startup — NOT per-request, it's reusable
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, features: dict):
        # Initialize all one-hot features to 0
        encoded = {col: 0 for col in self.model_feature_names}

        # Map numeric fields
        encoded['SeniorCitizen'] = int(features['SeniorCitizen'])
        encoded['tenure'] = int(features['tenure'])
        encoded['MonthlyCharges'] = float(features['MonthlyCharges'])
        encoded['TotalCharges'] = float(features['TotalCharges'])

        # Map categorical fields to their corresponding one-hot columns
        cat_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]

        for col in cat_cols:
            val = features.get(col)
            if val is not None:
                one_hot_col = f"{col}_{val}"
                if one_hot_col in encoded:
                    encoded[one_hot_col] = 1

        X = pd.DataFrame([encoded])[self.model_feature_names]

        prob = float(self.model.predict_proba(X)[:, 1][0])
        will_churn = prob >= self.threshold

        # Compute SHAP values for this single customer
        shap_vals = self.explainer.shap_values(X)[0]

        top_idx = np.argsort(np.abs(shap_vals))[::-1][:3]
        reasons = []
        for i in top_idx:
            feat_name = self.model_feature_names[i]
            reasons.append({
                'feature': feat_name,
                'value': float(X.iloc[0, i]),
                'impact': round(float(shap_vals[i]), 4),
                'direction': 'increases' if shap_vals[i] > 0 else 'decreases'
            })

        # Calculate costs based on customer profile and config
        fn_cost_if_missed = float(features['MonthlyCharges'] * self.fn_cost_multiplier)
        fp_cost_if_wrong = float(self.fp_cost)

        return {
            'churn_probability': round(prob, 4),
            'will_churn_flag': bool(will_churn),
            'threshold_used': self.threshold,
            'fn_cost_if_missed': fn_cost_if_missed,
            'fp_cost_if_wrong': fp_cost_if_wrong,
            'top_reasons': reasons
        }