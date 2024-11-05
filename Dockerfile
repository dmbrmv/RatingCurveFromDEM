FROM  nvidia/cuda:11.8.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV YOUR_ENV=flood \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100
WORKDIR /app

RUN apt-get update -y &&  apt-get upgrade -y
RUN  apt-get install software-properties-common -y
RUN  add-apt-repository ppa:deadsnakes/ppa -y
RUN yes 1 | apt-get update && apt-get -y install tzdata
RUN apt-get update -qq && \
    apt-get install --yes --quiet --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3.10-lib2to3 \
    python3.10-gdbm \
    python3.10-tk \
    curl \
    gcc \
    g++ \
    git \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    zsh \
    libspatialindex-dev \
    libcairo2-dev \
    libjpeg-dev \
    libgif-dev \
    vim \
    git-lfs \
    fontconfig \
    fonts-liberation \
    msttcorefonts \
    screen \
    --fix-missing -qq

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python

RUN apt list --installed | grep "gdal"

COPY  . .

ENV DATA_DIR=/app/data
RUN mkdir "${DATA_DIR}"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm /usr/bin/python3  && ln -s /usr/bin/python3.10 /usr/bin/python3
RUN rm ~/.cache/matplotlib -rf

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN zsh

