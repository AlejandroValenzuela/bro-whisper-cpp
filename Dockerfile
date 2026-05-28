FROM ubuntu:22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    python3 \
    python3-pip \
    curl \
    wget \
    ffmpeg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ggerganov/whisper.cpp.git /whisper.cpp && \
    cd /whisper.cpp && \
    cmake -B build && \
    cmake --build build --config Release

RUN cd /whisper.cpp/models && \
    bash download-ggml-model.sh tiny

RUN pip3 install --no-cache-dir \
    flask==2.3.3 \
    python-multipart==0.0.6 \
    gunicorn==21.2.0 \
    werkzeug==2.3.7

RUN mkdir -p /app/temp_audio

COPY whisper_cpp_api.py /app/whisper_cpp_api.py

RUN chmod +x /app/whisper_cpp_api.py

EXPOSE 5002

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

CMD ["gunicorn", \
     "--bind", "0.0.0.0:5002", \
     "--workers", "2", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "whisper_cpp_api:app"]
