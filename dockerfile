# ========================================================================
# CoordiNet - Social Coordination Detection Pipeline
# Dockerfile for Reproducible Execution
# ========================================================================

# Use a lightweight, slim Python 3.9 base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for numpy, scikit-learn, and networkx
# (minimal set for a slim image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# (Excludes .git, __pycache__, and other unnecessary files via .dockerignore)
COPY . .

# Create a non-root user to run the application (security best practice)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Set the default command to show help (optional)
CMD ["python", "-c", "print('CoordiNet container is ready. Run: python scripts/run.py --tier eval --threshold 97.5')"]

#i run into an input/output error when i run the docker build command.
#this is sometimes due to insufficient disk space or file system issues. Check your disk space and ensure you have enough space to build the Docker image. You can also try cleaning up unused Docker images and containers using `docker system prune`.