FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_BASE_URL="<your-active-api-base-url>"
ENV MODEL_NAME="<your-active-model-name>"
ENV HF_TOKEN=""
ENV ENV_URL="http://localhost:7860"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
