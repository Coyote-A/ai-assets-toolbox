FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git wget curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Qwen3-VL-2B-Instruct model using huggingface_hub for reliability
# (avoids wget rate-limiting and partial-download failures during Docker build)
RUN pip install --no-cache-dir huggingface_hub && \
    python3 -c "\
from huggingface_hub import snapshot_download; \
print('Downloading Qwen3-VL-2B-Instruct...'); \
snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='/app/models/qwen3-vl-2b'); \
print('Qwen3-VL-2B-Instruct downloaded successfully') \
"

# Copy application code (last for fast rebuilds)
COPY . /app/

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
