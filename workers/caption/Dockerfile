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

# Download Qwen3-VL-2B-Instruct model
# Single safetensors file (~5GB), no shards
RUN mkdir -p /app/models/qwen3-vl-2b && \
    echo "Downloading Qwen3-VL-2B-Instruct..." && \
    cd /app/models/qwen3-vl-2b && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/generation_config.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/preprocessor_config.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/video_preprocessor_config.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/vocab.json" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/merges.txt" && \
    wget -q "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json" && \
    wget "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors" && \
    echo "Qwen3-VL-2B-Instruct downloaded successfully"

# Copy application code (last for fast rebuilds)
COPY . /app/

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
