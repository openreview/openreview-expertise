FROM ubuntu:22.04

WORKDIR /app

ENV PYTHON_VERSION=3.8

ENV HOME="/app"

ENV PATH="/app/miniconda/bin:${PATH}"
ARG PATH="/app/miniconda/bin:${PATH}"

COPY . /app/openreview-expertise

RUN apt update \
    && apt install -y wget \
    && apt install -y make gcc \
    && apt install -y curl \
    && apt install -y build-essential \
    && apt install -y git \
    && apt install -y sudo \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt install -y tzdata \
    && cd $HOME \
    && wget "https://repo.anaconda.com/miniconda/Miniconda3-py38_22.11.1-1-Linux-x86_64.sh" -O miniconda.sh \
    && printf '%s' "473e5ecc8e078e9ef89355fbca21f8eefa5f9081544befca99867c7beac3150d  miniconda.sh" | sha256sum -c \
    && bash miniconda.sh -b -p $HOME/miniconda \
    && conda update -y conda \
    && conda create -n expertise python=$PYTHON_VERSION -c conda-forge

RUN echo "source ${HOME}/miniconda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc \
    && echo "conda activate expertise" >> ${HOME}/.bashrc \
    && /bin/bash -c "source ${HOME}/miniconda/etc/profile.d/conda.sh && conda activate expertise" \
    && python --version \
    && mkdir ${HOME}/expertise-utils \
    && cd ${HOME}/expertise-utils \
    && git clone https://github.com/allenai/specter.git \
    && cd specter \
    && wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz \
    && tar -xzvf archive.tar.gz \
    && conda install -y pytorch pytorch-cuda=11 -c pytorch -c nvidia  \
    && pip install -r requirements.txt \
    && python setup.py install \
    && conda install -y filelock \
    && cd .. \
    && wget https://storage.googleapis.com/openreview-public/openreview-expertise/models-data/multifacet_recommender_data.tar.gz -O mfr.tar.gz \
    && tar -xzvf mfr.tar.gz \
    && mv ./multifacet_recommender_data ./multifacet_recommender \
    && cd ${HOME}/openreview-expertise \
    && cp ${HOME}/openreview-expertise/expertise/service/config/default_gpu.cfg ${HOME}/openreview-expertise/expertise/service/config/production.cfg \
    && pip install -e . \
    && conda install -y intel-openmp \
    && conda install -y faiss-cpu -c pytorch \
    && pip install -I protobuf==3.20.1 \
    && pip install numpy==1.24.4 --force-reinstall

EXPOSE 8080

CMD ["FLASK_ENV=production", "python", "-m", "expertise.service", "--host", "0.0.0.0", "--port", "8080"]