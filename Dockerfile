# 构建阶段
FROM ascendai/cann:8.0.rc1-910b-ubuntu22.04-py3.8-linuxarm64 AS builder

RUN apt update && apt install -y --no-install-recommends \
        libsndfile1 ffmpeg git wget && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
        "numpy<2.0" \
        torch==2.1.0 \
        torch-npu==2.1.0.post10 \
        funasr \
        modelscope

# 运行阶段，只拷贝必要文件
FROM ascendai/cann:8.0.rc1-910b-ubuntu22.04-py3.8-linuxarm64

COPY --from=builder /usr/local/lib/python3.8/dist-packages /usr/local/lib/python3.8/dist-packages
COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /usr/local/bin/ffprobe /usr/local/bin/ffprobe

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace:${PYTHONPATH}

RUN mkdir -p /workspace/models
ENV MODELSCOPE_CACHE=/workspace/models
ENV FUNASR_CACHE_DIR=/workspace/models

WORKDIR /workspace
CMD ["npu-smi", "info"]
