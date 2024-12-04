FROM nvidia/cuda:11.6.1-base-ubuntu20.04

WORKDIR /app

ENV PYTHON_VERSION=3.9

ENV HOME="/app"

ENV PATH="/app/miniconda/bin:${PATH}"
ARG PATH="/app/miniconda/bin:${PATH}"

# Set the environment variable
ENV FLASK_ENV=production
ENV AIP_STORAGE_URI="gs://openreview-expertise/expertise-utils/"
ENV SPECTER_DIR="/app/expertise-utils/specter/"
ENV MFR_VOCAB_DIR="/app/expertise-utils/multifacet_recommender/feature_vocab_file"
ENV MFR_CHECKPOINT_DIR="/app/expertise-utils/multifacet_recommender/mfr_model_checkpoint/"

COPY . /app/openreview-expertise

RUN apt update \
    && apt install -y wget \
    && apt install -y make gcc \
    && apt install -y curl \
    && apt install -y build-essential \
    && apt install -y git \
    && apt install -y sudo \
    && apt install -y vim \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata \
    && cd $HOME \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-py39_24.9.2-0-Linux-x86_64.sh" -O miniconda.sh \
    && printf '%s' "4b540d78e5bdd770b39216c0563424ef6656504cbe24c67b2d0454c2eb7afe93  miniconda.sh" | sha256sum -c \
    && bash miniconda.sh -b -p $HOME/miniconda \
    && conda update -y conda \
    && conda create -n expertise python=$PYTHON_VERSION -c conda-forge

RUN echo "source ${HOME}/miniconda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc \
    && echo "conda activate expertise" >> ${HOME}/.bashrc \
    && /bin/bash -c "source ${HOME}/miniconda/etc/profile.d/conda.sh && conda activate expertise" \
    && python --version \
    && conda install -y pytorch pytorch-cuda=11 -c pytorch -c nvidia  \
    && mkdir ${HOME}/expertise-utils \
    && conda install -y filelock \
    && cd ${HOME}/openreview-expertise \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_container.cfg ${HOME}/openreview-expertise/expertise/service/config/production.cfg \
    && pip install -e . \
    && conda install -y intel-openmp \
    && conda install -y faiss-cpu -c pytorch \
    && pip install -I protobuf==3.20.1 \
    && pip install numpy==1.24.4 --force-reinstall

EXPOSE 8080

# Define the entry point and pass arguments separately
ENTRYPOINT ["python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080", "--container"]
