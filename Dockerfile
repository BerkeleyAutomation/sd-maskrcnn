FROM nvidia/cudagl:11.1.1-devel-ubuntu20.04

# env variables for tzdata install
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Vancouver

RUN apt-get update -y && \
    apt-get install -y \
        wget \
        software-properties-common \
        freeglut3-dev \
        liboctomap-dev \
        libfcl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3.8-dev python3-pip

# Install VHACD binary for convex decomp
RUN set -xe
RUN wget https://github.com/mikedh/v-hacd-1/raw/master/bin/linux/testVHACD
RUN echo "e1e79b2c1b274a39950ffc48807ecb0c81a2192e7d0993c686da90bd33985130  testVHACD" | sha256sum --check
RUN chmod +x testVHACD
RUN mv testVHACD /usr/bin/

# Install all other python deps
RUN mkdir -p sd-maskrcnn
COPY [ "./maskrcnn",  "sd-maskrcnn/maskrcnn" ]
COPY [ "./setup.py",  "sd-maskrcnn/setup.py" ]
RUN python3.8 -m pip install --no-cache-dir `python3.8 sd-maskrcnn/setup.py --list-setup`
RUN python3.8 -m pip install --no-cache-dir `python3.8 sd-maskrcnn/setup.py --list-train`

# Install repo
COPY [ ".",  "sd-maskrcnn/" ]
RUN python3.8 -m pip install --no-cache-dir sd-maskrcnn/

# Run generation
WORKDIR sd-maskrcnn
CMD python3.8 tools/train.py --config /cfg/train.yaml
