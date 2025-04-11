# ---- Stage 1: Build dependencies on Alpine ----
    FROM python:3.8.10-slim-buster AS builder

    # Install build dependencies for Alpine
    WORKDIR /install

    # Install build dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install minimal dependencies first
    RUN pip install --no-cache-dir --prefix=/install \
        flask==3.0.3 \
        Flask-Cors==5.0.0 \
        python-dotenv==1.0.1 \
        groq==0.20.0 \
        langchain==0.2.17 \
        langchain-core==0.2.43 \
        langchain-groq==0.1.10 \
        langchain-community==0.2.19 \
        langchain-text-splitters==0.2.4 \
        docx2txt==0.9 \
        faiss-cpu==1.8.0.post1 \
        torch==2.4.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu \
        sentence-transformers==3.2.1
    
    # Aggressive cleanup
    RUN find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    RUN find /install -name "*.pyc" -delete
    RUN find /install -name "*.pyo" -delete
    RUN find /install -name "*.c" -delete
    RUN find /install -name "*.cpp" -delete
    RUN find /install -name "*.h" -delete 2>/dev/null || true
    RUN find /install -name "*.html" -delete
    RUN find /install -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true
    RUN find /install -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
    RUN find /install -name "test" -type d -exec rm -rf {} + 2>/dev/null || true
    RUN find /install -name "examples" -type d -exec rm -rf {} + 2>/dev/null || true
    RUN find /install -name "docs" -type d -exec rm -rf {} + 2>/dev/null || true

    # Clean up TensorFlow
    RUN rm -rf /install/lib/python3.8/site-packages/tensorflow/include 2>/dev/null || true
    RUN rm -rf /install/lib/python3.8/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so 2>/dev/null || true
    RUN find /install/lib/python3.8/site-packages/tensorflow/ -name "*test*" -type d -exec rm -rf {} + 2>/dev/null || true
    RUN find /install/lib/python3.8/site-packages/tensorflow/ -name "*gpu*" -type d -exec rm -rf {} + 2>/dev/null || true

    
    # ---- Stage 2: Runtime image ----
    FROM python:3.8.10-slim-buster

    # Copy installed packages from builder stage
    COPY --from=builder /install /usr/local
        
    # Set working directory
    WORKDIR /chatbot_lite
        
    # Set environment variables
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    ENV OMP_NUM_THREADS=1
    ENV TF_CPP_MIN_LOG_LEVEL=3
        
        # Copy environment file and application code
    COPY .env .env
    COPY app/ app/
        
        # Set working directory to the app folder
    WORKDIR /chatbot_lite/app
        
        # Expose port
    EXPOSE 5000
        
        # Command to run the application
    CMD ["python", "app.py"]