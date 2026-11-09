FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY mmm ./mmm
RUN pip install --no-cache-dir -e ".[api,dashboard]"
COPY app.py api.py ./
EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
