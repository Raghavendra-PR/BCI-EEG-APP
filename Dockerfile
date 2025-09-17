FROM python:3.10-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies (libgomp1 needed by lightgbm wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (leverage Docker layer caching)
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Copy application code (models, data samples, and app)
COPY . .

# Streamlit runs on 8501 by default
EXPOSE 8501

# Ensure Streamlit binds to all interfaces inside the container
CMD ["python", "-m", "streamlit", "run", "eeg_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]


