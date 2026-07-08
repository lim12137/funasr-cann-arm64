# 构建阶段：只负责装 Python 依赖，装到已知目录
FROM ascendai/cann:8.0.rc1-910b-ubuntu22.04-py3.8 AS builder

RUN apt update && apt install -y --no-install-recommends git wget && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir --target=/opt/pylibs \
        "numpy<2.0" \
        torch==2.1.0 \
        torch-npu==2.1.0.post10 \
        funasr \
        modelscope

# 运行阶段
FROM ascendai/cann:8.0.rc1-910b-ubuntu22.04-py3.8

# 运行时真正需要的系统库（ffmpeg/libsndfile 在推理时要用）
RUN apt update && apt install -y --no-install-recommends libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 拷贝构建阶段装好的 Python 依赖
COPY --from=builder /opt/pylibs /opt/pylibs

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/opt/pylibs:${PYTHONPATH}

RUN mkdir -p /workspace/models
ENV MODELSCOPE_CACHE=/workspace/models
ENV FUNASR_CACHE_DIR=/workspace/models

WORKDIR /workspace
CMD ["npu-smi", "info"]
