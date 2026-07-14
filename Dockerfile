FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/

# Model artifacts — must be copied from your notebook's output before building:
#   churn_model_final.json, model_config.json, feature_names.json
COPY churn_model_final.json .
COPY model_config.json .
COPY feature_names.json .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]