FROM ubuntu:16.04
# LABEL maintainer fabio.carrara@isti.cnr.it

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libopenblas-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-setuptools \
        python-scipy && \    
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/opt/caffe
RUN mkdir -p $CAFFE_ROOT && cd $CAFFE_ROOT && \
    git init && git remote add origin https://github.com/BVLC/caffe && \
    git fetch origin pull/4163/head:pr4163 && git checkout pr4163    
WORKDIR $CAFFE_ROOT

RUN pip install --upgrade pip && cd python && for req in $(cat requirements.txt) pydot; do python -m pip install --no-cache $req; done && cd .. && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 -DBLAS=Open .. && \
    make -j"$(nproc)"

RUN python -m pip install --no-cache opencv-python dask toolz tqdm

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
RUN mkdir /code
WORKDIR /code

RUN wget http://download.europe.naverlabs.com/Computer-Vision-CodeandModels/deep_retrieval.tgz && \
    tar xf deep_retrieval.tgz \
        deep_retrieval/custom_layers.py \
        deep_retrieval/model.caffemodel \
        deep_retrieval/deploy_resnet101.prototxt \
        deep_retrieval/deploy_resnet101_normpython.prototxt && \
    mv deep_retrieval net && mv net/custom_layers.py . && rm deep_retrieval.tgz

RUN pip install tornado flask-restful
ADD . /code

ENTRYPOINT [ "python" ]
CMD [ "service.py" ]
