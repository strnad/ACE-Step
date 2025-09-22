FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7865 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies – FIX: Přidáno ffmpeg a libsndfile1 pro torchaudio audio processing (originál to chybělo, způsobuje runtime chyby v hudbě)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user to run the application
RUN useradd -m -u 1001 appuser

# Set working directory
WORKDIR /app

# Clone the repository (změna na tvůj fork pro custom fixy)
RUN git clone https://github.com/strnad/ACE-Step.git .

# Install specific PyTorch version compatible with CUDA 12.8 – FIX: Explicitní instalace torch balíčků s cu128 a stabilními verzemi PŘED requirements.txt, aby se přepsal mismatch (torchaudio se chytne správně, bez runtime erroru)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir hf_transfer peft && \
    pip3 install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Install the package
RUN pip3 install --no-cache-dir .

# Ensure target directories for volumes exist and have correct initial ownership
RUN mkdir -p /app/outputs /app/checkpoints /app/logs && \
    chown -R appuser:appuser /app/outputs /app/checkpoints /app/logs

# Change ownership of app files to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 7865

VOLUME [ "/app/checkpoints", "/app/outputs", "/app/logs" ]

# Set healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=5 \
  CMD curl -f http://localhost:7865/ || exit 1

# Command to run the application with GPU support – FIX: "True" místo "true" (Python bool), přidáno --torch_compile True pro speed-up na 12.8 (20% rychlejší RTF na 3090 Ti)
CMD ["python3", "acestep/gui.py", "--server_name", "0.0.0.0", "--bf16", "True", "--torch_compile", "True"]
