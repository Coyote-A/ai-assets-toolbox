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
COPY workers/caption/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Qwen3-VL-2B-Instruct model
# Single safetensors file (~4.26 GB), no shards
RUN mkdir -p /app/models/qwen3-vl-2b && \
    echo "Downloading Qwen3-VL-2B-Instruct..." && \
    cd /app/models/qwen3-vl-2b && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/config.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/generation_config.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/preprocessor_config.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/video_preprocessor_config.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer_config.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/tokenizer.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/vocab.json" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/merges.txt" && \
    wget -q --tries=3 "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/chat_template.json" && \
    wget --tries=3 --progress=dot:giga "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/resolve/main/model.safetensors" && \
    echo "Verifying model.safetensors size..." && \
    ACTUAL=$(stat -c%s model.safetensors) && \
    EXPECTED=4572213416 && \
    if [ "$ACTUAL" -lt "$EXPECTED" ]; then \
        echo "ERROR: model.safetensors is too small ($ACTUAL bytes, expected ~$EXPECTED). Download may be incomplete." && \
        exit 1; \
    fi && \
    echo "Qwen3-VL-2B-Instruct downloaded successfully ($(du -sh model.safetensors | cut -f1))"

# Copy application code (last for fast rebuilds)
COPY workers/caption/ /app/

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
