# Stage 1: Builder stage - install build tools and compile packages
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ARG OPENREVIEW_PY_VERSION=master

WORKDIR /app

ENV PYTHON_VERSION=3.11 \
    HOME="/app" \
    FLASK_ENV=production \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    wget \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN wget -O - https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /root/.cache/pip

# Copy only requirements first for better caching
COPY setup.py /app/openreview-expertise/

# Install PyTorch with CUDA support first (large package, benefits from caching)
RUN pip3.11 install --no-cache-dir --upgrade pip setuptools wheel \
    && pip3.11 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && rm -rf /root/.cache/pip

# Install faiss-cpu
RUN pip3.11 install --no-cache-dir faiss-cpu \
    && rm -rf /root/.cache/pip

# Copy the rest of the application
COPY . /app/openreview-expertise

# Install the application and its dependencies
RUN cd /app/openreview-expertise \
    && pip3.11 install --no-cache-dir -e . \
    && pip3.11 install --no-cache-dir -I protobuf==3.20.1 \
    && pip3.11 install --no-cache-dir -e "git+https://github.com/openreview/openreview-py.git@${OPENREVIEW_PY_VERSION}#egg=openreview-py" \
    && rm -rf /root/.cache/pip

# Stage 2: Runtime stage - minimal image with only runtime dependencies
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

WORKDIR /app

ENV PYTHON_VERSION=3.11 \
    HOME="/app" \
    FLASK_ENV=production \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/" \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:$PATH"

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    python3.11 \
    python3.11-distutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install pip for Python 3.11
RUN wget -O - https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && rm -rf /root/.cache/pip

# Copy Python packages and site-packages from builder stage
# Note: pip installs packages to /usr/local/lib/python3.11/dist-packages when installed system-wide
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/openreview-expertise /app/openreview-expertise

# Create expertise-utils directory and copy config files
RUN mkdir -p ${HOME}/expertise-utils \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_container.cfg \
       ${HOME}/openreview-expertise/expertise/service/config/production.cfg

EXPOSE 8080

ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
