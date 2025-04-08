# ---- Stage 1: Build Environment ----
    FROM python:3.8.10-slim-buster AS builder

    # Set working directory
    WORKDIR /app
    
    # Copy and install dependencies
    #COPY requirements.txt .
    #RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
    # Manually install dependencies
    RUN pip install --no-cache-dir aiohappyeyeballs==2.4.4
    RUN pip install --no-cache-dir aiohttp==3.10.1
    RUN pip install --no-cache-dir aiosignal==1.3.1
    RUN pip install --no-cache-dir annotated-types==0.7.0
    RUN pip install --no-cache-dir anyio==4.5.2
    RUN pip install --no-cache-dir async-timeout==4.0.3
    RUN pip install --no-cache-dir attrs==25.3.0
    RUN pip install --no-cache-dir blinker==1.8.2
    RUN pip install --no-cache-dir certifi==2025.1.31
    RUN pip install --no-cache-dir charset-normalizer==3.4.1
    RUN pip install --no-cache-dir click==8.1.8
    RUN pip install --no-cache-dir colorama==0.4.6
    RUN pip install --no-cache-dir dataclasses-json==0.6.7
    RUN pip install --no-cache-dir distro==1.9.0
    RUN pip install --no-cache-dir docx2txt==0.9
    RUN pip install --no-cache-dir exceptiongroup==1.2.2
    RUN pip install --no-cache-dir faiss-cpu==1.8.0.post1
    RUN pip install --no-cache-dir filelock==3.16.1
    RUN pip install --no-cache-dir flask==3.0.3
    RUN pip install --no-cache-dir Flask-Cors==5.0.0
    RUN pip install --no-cache-dir frozenlist==1.5.0
    RUN pip install --no-cache-dir fsspec==2025.3.0
    RUN pip install --no-cache-dir greenlet==3.1.1
    RUN pip install --no-cache-dir groq==0.20.0
    RUN pip install --no-cache-dir h11==0.14.0
    RUN pip install --no-cache-dir httpcore==1.0.7
    RUN pip install --no-cache-dir httpx==0.28.1
    RUN pip install --no-cache-dir huggingface-hub==0.29.3
    RUN pip install --no-cache-dir idna==3.10
    RUN pip install --no-cache-dir importlib-metadata==8.5.0
    RUN pip install --no-cache-dir itsdangerous==2.2.0
    RUN pip install --no-cache-dir jinja2==3.1.6
    RUN pip install --no-cache-dir joblib==1.4.2
    RUN pip install --no-cache-dir jsonpatch==1.33
    RUN pip install --no-cache-dir jsonpointer==3.0.0
    RUN pip install --no-cache-dir langchain==0.2.17
    RUN pip install --no-cache-dir langchain-community==0.2.19
    RUN pip install --no-cache-dir langchain-core==0.2.43
    RUN pip install --no-cache-dir langchain-groq==0.1.10
    RUN pip install --no-cache-dir langchain-text-splitters==0.2.4
    RUN pip install --no-cache-dir langsmith==0.1.147
    RUN pip install --no-cache-dir MarkupSafe==2.1.5
    RUN pip install --no-cache-dir marshmallow==3.22.0
    RUN pip install --no-cache-dir mpmath==1.3.0
    RUN pip install --no-cache-dir multidict==6.1.0
    RUN pip install --no-cache-dir mypy-extensions==1.0.0
    RUN pip install --no-cache-dir networkx==3.1
    RUN pip install --no-cache-dir numpy==1.24.4
    RUN pip install --no-cache-dir orjson==3.10.15
    RUN pip install --no-cache-dir packaging==24.2
    RUN pip install --no-cache-dir pillow==10.4.0
    RUN pip install --no-cache-dir propcache==0.2.0
    RUN pip install --no-cache-dir pydantic==2.10.6
    RUN pip install --no-cache-dir pydantic-core==2.27.2
    RUN pip install --no-cache-dir python-dotenv==1.0.1
    RUN pip install --no-cache-dir PyYAML==6.0.2
    RUN pip install --no-cache-dir regex==2024.11.6
    RUN pip install --no-cache-dir requests==2.32.3
    RUN pip install --no-cache-dir requests-toolbelt==1.0.0
    RUN pip install --no-cache-dir safetensors==0.5.3
    RUN pip install --no-cache-dir scikit-learn==1.3.2
    RUN pip install --no-cache-dir scipy==1.10.1
    RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
    RUN pip install --no-cache-dir sentence-transformers==3.2.1
    RUN pip install --no-cache-dir sniffio==1.3.1
    RUN pip install --no-cache-dir sqlalchemy==2.0.40
    RUN pip install --no-cache-dir sympy==1.13.3
    RUN pip install --no-cache-dir tenacity==8.5.0
    RUN pip install --no-cache-dir threadpoolctl==3.5.0
    RUN pip install --no-cache-dir tokenizers==0.20.3
    #RUN pip install --no-cache-dir torch==2.4.1
    RUN pip install --no-cache-dir tqdm==4.67.1
    RUN pip install --no-cache-dir transformers==4.46.3
    RUN pip install --no-cache-dir typing-extensions==4.13.0
    RUN pip install --no-cache-dir typing-inspect==0.9.0
    RUN pip install --no-cache-dir urllib3==2.2.3
    RUN pip install --no-cache-dir werkzeug==3.0.6
    RUN pip install --no-cache-dir yarl==1.15.2
    RUN pip install --no-cache-dir zipp==3.20.2
    # ---- Stage 2: Production Image ----
    #FROM python:3.8.10
    
    # ---- Stage 2: Production Image ----
    FROM python:3.8.10-slim-buster

    # Set working directory
    WORKDIR /app

    # Copy only the installed dependencies from the builder stage
    COPY --from=builder /usr/local /usr/local

    # Copy application source code from the 'app' subdirectory
    COPY app .

    # Expose port 5000
    EXPOSE 5000

    # Command to run the application
    CMD ["python", "app.py"]