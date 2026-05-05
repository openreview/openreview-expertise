FROM python:3.11-slim-bookworm AS builder

ENV PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV="/app/venv" \
    PATH="/app/venv/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/openreview-expertise

RUN python -m venv ${VIRTUAL_ENV} \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir filelock faiss-cpu \
    && pip install --no-cache-dir -e /app/openreview-expertise


FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV HOME="/app" \
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

# Runtime libs the copied CPython 3.11 binary dynamically links against
# (built on Debian bookworm; these are the Ubuntu 24.04 equivalents).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libssl3 \
        libffi8 \
        libsqlite3-0 \
        libreadline8 \
        liblzma5 \
        libbz2-1.0 \
        libncursesw6 \
        libuuid1 \
        libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/python3.11 /usr/local/bin/python3.11
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/lib/libpython3.11.so /usr/local/lib/libpython3.11.so
COPY --from=builder /usr/local/lib/libpython3.11.so.1.0 /usr/local/lib/libpython3.11.so.1.0
COPY --from=builder /app/venv /app/venv
COPY --from=builder /app/openreview-expertise /app/openreview-expertise

RUN ln -s /usr/local/bin/python3.11 /usr/local/bin/python3 \
    && ldconfig

# HuggingFace models (specter, specter2 base + adapter, scincl) are not baked into
# the image — they are fetched from gs://openreview-expertise/expertise-utils/hf_models/
# at job start, selectively per-request, by expertise.service.load_model_artifacts.

RUN mkdir ${HOME}/expertise-utils \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_container.cfg \
       ${HOME}/openreview-expertise/expertise/service/config/production.cfg

EXPOSE 8080

ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
