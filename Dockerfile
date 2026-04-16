FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV PYTHON_VERSION=3.11 \
    HOME="/app" \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV="/app/venv" \
    PATH="/app/venv/bin:${PATH}" \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/" \
    SPECTER_HF_DIR="/app/expertise-utils/hf_models/specter" \
    SPECTER2_HF_DIR="/app/expertise-utils/hf_models/specter2_base" \
    SPECTER2_ADAPTER_DIR="/app/expertise-utils/hf_models/specter2_adapter" \
    SCINCL_HF_DIR="/app/expertise-utils/hf_models/scincl"

COPY . /app/openreview-expertise

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        build-essential \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/* \
    \
    && python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV} \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir filelock intel-openmp faiss-cpu \
    && pip install --no-cache-dir -e /app/openreview-expertise \
    && pip install --no-cache-dir -I protobuf==3.20.1 \
    && apt-get purge -y build-essential software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace models (specter, specter2 base + adapter, scincl) are not baked into
# the image — they are fetched from gs://openreview-expertise/expertise-utils/hf_models/
# at job start, selectively per-request, by expertise.service.load_model_artifacts.

RUN mkdir ${HOME}/expertise-utils \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_container.cfg \
       ${HOME}/openreview-expertise/expertise/service/config/production.cfg

EXPOSE 8080

ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
