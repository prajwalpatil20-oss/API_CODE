# Base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Railway automatically sets PORT environment variable
# Expose it (optional but good practice)
EXPOSE 8000

# Run FastAPI with uvicorn, binding to Railway's dynamic PORT
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
