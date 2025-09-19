# GovDocShield X Dockerfile
# Multi-stage build for quantum-neuromorphic threat detection

# Base stage with Python and quantum dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements files
COPY requirements.txt requirements-api.txt requirements-cli.txt requirements-dashboard.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY govdocshield.py dashboard.py ./
COPY config/ ./config/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash govdocshield
RUN chown -R govdocshield:govdocshield /app
USER govdocshield

# Expose ports
EXPOSE 8080 50051 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8080"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-asyncio black isort mypy

# Copy application code
COPY . .

# Expose additional ports for development
EXPOSE 8080 50051 8501 8000

CMD ["python", "-m", "uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# CLI stage
FROM base as cli

# Install CLI dependencies
RUN pip install --no-cache-dir -r requirements-cli.txt

# Copy CLI components
COPY src/cli/ ./src/cli/
COPY src/shared/ ./src/shared/
COPY govdocshield.py ./

# Set CLI as default entrypoint
ENTRYPOINT ["python", "govdocshield.py"]

# Dashboard stage  
FROM base as dashboard

# Install dashboard dependencies
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# Copy dashboard components
COPY src/dashboard/ ./src/dashboard/
COPY src/shared/ ./src/shared/
COPY dashboard.py ./

# Expose Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]