# Credit: https://github.com/sshuair/dl-docker-geospatial/blob/master/dockerfiles/all-devel.dockerfile

ARG CUDA=11.4.0
ARG CUDNN=7

FROM nvidia/cuda-arm64:${CUDA}-runtime-ubuntu20.04

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ARG TORCH_VERSION=1.0.1.post2

RUN     rm /etc/apt/sources.list.d/cuda.list \
    &&  rm /etc/apt/sources.list.d/nvidia-ml.list \
    &&  apt-get update --fix-missing \
    &&  apt-get install -y --no-install-recommends \
            build-essential software-properties-common \
            gfortran libffi-dev build-essential libfreetype6-dev libpng-dev libzmq3-dev libsm6 \
            python3 python3-dev python3-tk python3-pip \
            libspatialindex-dev gdal-bin libgdal-dev \
            vim wget zip \
    &&  apt-get clean \
    &&  rm -rf /var/src/apt/lists/*

RUN     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
    &&  bash Miniconda3-latest-Linux-aarch64.sh -b \
    &&  rm -f Miniconda3-latest-Linux-aarch64.sh \
    &&  . ~/miniconda3/bin/activate \
    &&  conda install llvmdev

RUN     pip3 install --upgrade pip \
    &&  pip3 --no-cache-dir install --upgrade --prefer-binary \
            setuptools wheel cython packaging pyclean \
    &&  pip3 install --no-cache-dir --upgrade --prefer-binary \
            jupyterlab numpy torch \
            Pillow matplotlib opencv-python-headless \
            'rasterio<1.3' tifffile \
            scikit-image

WORKDIR /workspace/lib

COPY ./requirements.txt ./

RUN pip3 install --prefer-binary -r requirements.txt

COPY src/* ./

RUN pwd && pyclean .

#RUN pip install -e .
ENV PATH=$PATH:/workspace/lib/src
ENV PYTHONPATH="/workspace/lib/src"
#RUN export PYTHONPATH=/workspace/src/src && python3 entrypoint.py download
