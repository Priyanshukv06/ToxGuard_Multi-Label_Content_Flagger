FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy only what's needed for inference
COPY app/ ./app/
COPY models/ ./models/
COPY data_sample/ ./data_sample/

# Expose port
EXPOSE 8000

# Run with uvicorn (single worker for free tier memory constraints)
# Use shell form so the $PORT environment variable can be evaluated (Render sets this dynamically)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
