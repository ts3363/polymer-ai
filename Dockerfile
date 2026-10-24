FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Install minimal OS dependencies for RDKit and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch to minimize image footprint and build duration
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY . .

# Expose port for Hugging Face Spaces (7860) or cloud providers
EXPOSE 7860

# Launch server
CMD ["python", "backend.py"]
