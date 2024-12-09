FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

WORKDIR /app

ENV PYTHON_VERSION=3.9 \
    HOME="/app" \
    PATH="/app/miniconda/bin:${PATH}" \
    FLASK_ENV=production \
    AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/" \
    SPECTER_DIR="/app/expertise-utils/specter/" \
    MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file" \
    MFR_CHECKPOINT_DIR="/app/expertise-utils/mfr_model_checkpoint/"

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
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh" -O miniconda.sh \
    && echo "4b540d78e5bdd770b39216c0563424ef6656504cbe24c67b2d0454c2eb7afe93  miniconda.sh" | sha256sum -c - \
    && bash miniconda.sh -b -p $HOME/miniconda \
    && rm miniconda.sh \
    \
    && conda update -y conda \
    && conda create -y -n expertise python=$PYTHON_VERSION -c conda-forge \
    \
    && . $HOME/miniconda/etc/profile.d/conda.sh \
    && conda activate expertise \
    && conda install -y pytorch pytorch-cuda=11 -c pytorch -c nvidia \
    && conda install -y filelock intel-openmp faiss-cpu -c pytorch \
    && pip install --no-cache-dir -e $HOME/openreview-expertise \
    && pip install --no-cache-dir -I protobuf==3.20.1 \
    && pip install --no-cache-dir numpy==1.24.4 --force-reinstall \
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
