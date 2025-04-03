# ---- Stage 1: Build Environment ----
    FROM python:3.8-slim AS builder

    # Set working directory
    WORKDIR /app
    
    # Install dependencies
    RUN apt-get update && apt-get install -y \
        gcc python3-dev libpq-dev && \
        rm -rf /var/lib/apt/lists/*
    
    # Copy only requirements first (better for caching)
    COPY requirements.txt .
    
    # Install dependencies into a temporary directory
    RUN pip install --no-cache-dir --target=/dependencies -r requirements.txt
    
    # ---- Stage 2: Minimal Runtime Image ----
    FROM python:3.8-slim
    
    # Set working directory
    WORKDIR /app
    
    # Copy dependencies from the builder stage
    COPY --from=builder /dependencies /usr/local/lib/python3.8/site-packages/
    
    # Copy application files (excluding unnecessary files)
    COPY . .
    
    # Expose the required port
    EXPOSE 5000
    
    # Run the application
    CMD ["python", "app.py"]
    