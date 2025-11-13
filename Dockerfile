FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG OPENREVIEW_PY_VERSION=master

WORKDIR /app

ENV PYTHON_VERSION=3.11 \
    HOME="/app" \
    CONDA_DIR="/app/miniconda" \
    PATH="/app/miniconda/bin:${PATH}" \
    FLASK_ENV=production \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/"

# Copy first so the whole install happens in a single layer that we can
# aggressively clean up at the end
COPY . /app/openreview-expertise

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        wget \
        curl \
        ca-certificates \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    \
    # Install Miniconda (Python 3.11)
    && cd "$HOME" \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh" -O miniconda.sh \
    && echo "62ef806265659c47e37e22e8f9adce29e75c4ea0497e619c280f54c823887c4f  miniconda.sh" | sha256sum -c - \
    && bash miniconda.sh -b -p "$CONDA_DIR" \
    && rm miniconda.sh \
    \
    # Use the *base* conda env directly (no duplicate env) to save disk
    && conda install -y filelock intel-openmp faiss-cpu -c pytorch \
    && conda install -y pytorch pytorch-cuda=12.4 -c pytorch -c nvidia \
    \
    # Install Python packages (no pip cache)
    && python -m pip install --no-cache-dir -e "$HOME/openreview-expertise" \
    && python -m pip install --no-cache-dir -I protobuf==3.20.1 \
    && python -m pip install --no-cache-dir \
         "git+https://github.com/openreview/openreview-py.git@${OPENREVIEW_PY_VERSION}#egg=openreview-py" \
    \
    # Clean up conda caches
    && conda clean --all -y \
    \
    # Remove build tools and APT cache in the *same layer* so they don't stay in the final image
    && apt-get purge -y build-essential wget curl git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# We now use the base conda environment's python, no extra PATH needed

RUN mkdir -p "${HOME}/expertise-utils" \
    && cp "${HOME}/openreview-expertise/expertise/service/config/default_container.cfg" \
          "${HOME}/openreview-expertise/expertise/service/config/production.cfg"

EXPOSE 8080

ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
