FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        ninja-build \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llm-lab
COPY . .

# The RunPod pipeline targets a single RTX 4090 (Ada, compute capability 8.9).
# Building one architecture makes image builds markedly faster than the portable
# CUDA default and does not affect checkpoint compatibility.
RUN cmake -S . -B build/cuda-release -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLM_LAB_ENABLE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=89 \
    && cmake --build build/cuda-release --parallel

COPY deploy/runpod/entrypoint.sh /usr/local/bin/llm-lab-runpod-train
RUN chmod 0755 /usr/local/bin/llm-lab-runpod-train

ENTRYPOINT ["/usr/local/bin/llm-lab-runpod-train"]
