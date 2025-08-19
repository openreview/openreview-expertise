FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ARG OPENREVIEW_PY_VERSION=latest

WORKDIR /app

ENV PYTHON_VERSION=3.11 \
    HOME="/app" \
    PATH="/app/miniconda/bin:${PATH}" \
    FLASK_ENV=production \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/"

COPY . /app/openreview-expertise

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    \
    && cd $HOME \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-py311_24.9.2-0-Linux-x86_64.sh" -O miniconda.sh \
    && echo "62ef806265659c47e37e22e8f9adce29e75c4ea0497e619c280f54c823887c4f  miniconda.sh" | sha256sum -c - \
    && bash miniconda.sh -b -p $HOME/miniconda \
    && rm miniconda.sh \
    \
    && conda update -y conda \
    && conda create -y -n expertise python=$PYTHON_VERSION -c conda-forge \
    \
    && . $HOME/miniconda/etc/profile.d/conda.sh \
    && conda activate expertise \
    && conda install -y filelock intel-openmp faiss-cpu -c pytorch \
    && conda install --force-reinstall pytorch pytorch-cuda=12.4 -c pytorch -c nvidia \
    && python -m pip install --no-cache-dir -e $HOME/openreview-expertise \
    && python -m pip install --no-cache-dir -I protobuf==3.20.1 \
    && if [ "${OPENREVIEW_PY_VERSION}" = "latest" ]; then \
        python -m pip install -e "git+https://github.com/openreview/openreview-py.git#egg=openreview-py"; \
    else \
        python -m pip install -e "git+https://github.com/openreview/openreview-py.git@${OPENREVIEW_PY_VERSION}#egg=openreview-py"; \
    fi \
    && conda clean --all -y \
    && apt-get purge -y build-essential wget curl git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Add conda environment bin to PATH so that 'python' uses the environment by default
ENV PATH="/app/miniconda/envs/expertise/bin:${PATH}"

RUN mkdir ${HOME}/expertise-utils \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_container.cfg \
       ${HOME}/openreview-expertise/expertise/service/config/production.cfg

EXPOSE 8080

ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
