# Multi-stage Dockerfile for AI Hardware Co-Design Playground
# Supports both development and production environments

# Base image with Python and Node.js
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    pkg-config \
    libpq-dev \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy dependency files
COPY package.json package-lock.json* ./
COPY requirements.txt requirements-dev.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Install Node.js dependencies
RUN npm ci --only=production

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt \
    && npm install

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for development
RUN useradd --create-home --shell /bin/bash --uid 1000 developer \
    && chown -R developer:developer /app

USER developer

# Set environment variables for development
ENV PYTHONPATH=/app/backend \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=development \
    FLASK_ENV=development

# Expose ports
EXPOSE 8000 3000 8888

# Development command
CMD ["npm", "run", "dev"]

# Production backend stage
FROM base as backend-production

# Copy backend source code
COPY backend/ ./backend/
COPY tests/ ./tests/

# Install package in production mode
RUN cd backend && pip install --no-cache-dir -e .

# Create non-root user for production
RUN useradd --create-home --shell /bin/bash --uid 1001 appuser \
    && chown -R appuser:appuser /app

USER appuser

# Set environment variables for production
ENV PYTHONPATH=/app/backend \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NODE_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production backend command
CMD ["uvicorn", "codesign_playground.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Production frontend stage
FROM node:18-alpine as frontend-production

WORKDIR /app

# Copy frontend source
COPY frontend/package*.json ./
COPY frontend/ ./

# Install dependencies and build
RUN npm ci --only=production \
    && npm run build

# Nginx stage for serving frontend
FROM nginx:alpine as frontend-nginx

# Copy built frontend
COPY --from=frontend-production /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Production frontend command
CMD ["nginx", "-g", "daemon off;"]

# Hardware simulation stage (with additional tools)
FROM base as hardware-simulation

# Install hardware simulation tools
RUN apt-get update && apt-get install -y \
    verilator \
    gtkwave \
    iverilog \
    yosys \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python packages for hardware simulation
RUN pip install --no-cache-dir \
    cocotb \
    cocotb-test \
    pyverilog \
    nmigen \
    litex

# Copy source code
COPY backend/ ./backend/
COPY hardware/ ./hardware/
COPY tests/ ./tests/

# Create hardware tools user
RUN useradd --create-home --shell /bin/bash --uid 1002 hwuser \
    && chown -R hwuser:hwuser /app

USER hwuser

# Set environment for hardware simulation
ENV PYTHONPATH=/app/backend:/app/hardware \
    COCOTB_REDUCED_LOG_FMT=1

# Expose ports for simulation and debugging
EXPOSE 8000 8888 5000

# Hardware simulation command
CMD ["python", "-m", "codesign_playground.simulation.server"]

# Jupyter notebook stage
FROM development as jupyter

# Install Jupyter and extensions
RUN pip install --no-cache-dir \
    jupyterlab \
    jupyter-lsp \
    jupyterlab-git \
    jupyterlab-code-formatter

# Create notebooks directory
RUN mkdir -p /app/notebooks

# Set Jupyter configuration
ENV JUPYTER_ENABLE_LAB=yes \
    JUPYTER_TOKEN=codesign-playground

# Expose Jupyter port
EXPOSE 8888

# Jupyter command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=codesign-playground"]

# All-in-one development stage
FROM development as all-in-one

# Install all optional dependencies
RUN pip install --no-cache-dir \
    torch \
    tensorflow \
    jupyterlab \
    cocotb

# Copy all source code
COPY . .

# Set comprehensive environment
ENV PYTHONPATH=/app/backend:/app/hardware \
    NODE_ENV=development

# Expose all ports
EXPOSE 3000 8000 8888 6006

# All-in-one command (using docker-compose services)
CMD ["npm", "run", "dev"]