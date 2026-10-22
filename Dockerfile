# ─────────────────────────────────────────────
# Stage 1: Builder — install heavy dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /install

# System deps needed to compile some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer cache — only rebuilds if requirements change)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install/deps --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: Runtime — lean final image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local

# Copy application source
COPY app.py .

# Copy model artifacts
COPY scaler_v2.pkl \
     features_v2.pkl \
     imputer_v2.pkl \
     best_threshold.pkl \
     voting_ensemble.pkl \
     gradient_boosting_model.pkl \
     xgboost_model.pkl \
     maintenance_model_v2.keras \
     model_metadata.json \
     ./

# Copy the large Random Forest model (optional — remove if using Git LFS)
# COPY random_forest_model.pkl .

# Copy logo assets (optional — app degrades gracefully if missing)
COPY Schindler.png Schindler_2.png ./

# Streamlit config
COPY .streamlit/config.toml .streamlit/config.toml

# SQLite database is created at runtime — just ensure the directory is writable
# Data is persisted via a Docker volume (see docker-compose.yml)

# Expose Streamlit default port
EXPOSE 8501

# Health check — hits Streamlit's built-in health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Non-root user for security
RUN useradd -m -u 1001 schindler && chown -R schindler:schindler /app
USER schindler

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
