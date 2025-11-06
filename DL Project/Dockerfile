# Multi-stage Dockerfile for Sentiment Analysis API
# Stage 1: Base Python environment with dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base as application

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/saved_model /app/logs /app/checkpoints /app/evaluation_results \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - start the API server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative entry points for different use cases:
# 
# Training:
# docker run -v $(pwd)/saved_model:/app/saved_model sentiment-analysis \
#   python main.py train --epochs 3
#
# API Server (default):
# docker run -p 8000:8000 -v $(pwd)/saved_model:/app/saved_model sentiment-analysis
#
# Evaluation:  
# docker run -v $(pwd)/saved_model:/app/saved_model sentiment-analysis \
#   python main.py evaluate
#
# Interactive shell:
# docker run -it -v $(pwd)/saved_model:/app/saved_model sentiment-analysis bash

# Multi-stage production build (optional, for smaller image)
# FROM python:3.9-slim as production
# 
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1
# 
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
# 
# RUN groupadd -r appuser && useradd -r -g appuser appuser
# 
# WORKDIR /app
# 
# # Copy only the installed packages from base stage
# COPY --from=base /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# COPY --from=base /usr/local/bin /usr/local/bin
# 
# # Copy application
# COPY --chown=appuser:appuser . .
# 
# RUN mkdir -p /app/saved_model /app/logs /app/checkpoints /app/evaluation_results \
#     && chown -R appuser:appuser /app
# 
# USER appuser
# 
# EXPOSE 8000
# 
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1
# 
# CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
