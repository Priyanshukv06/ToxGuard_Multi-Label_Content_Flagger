FROM python:3.10-slim

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Set home directory and path for the new user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements first for Docker layer caching
COPY requirements_backend.txt .
RUN pip install --no-cache-dir -r requirements_backend.txt

# Copy only what's needed for inference
COPY --chown=user app/ ./app/
COPY --chown=user models/ ./models/
COPY --chown=user data_sample/ ./data_sample/

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run with uvicorn
# We set the default port to 7860 for Hugging Face, but allow it to be overridden
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
