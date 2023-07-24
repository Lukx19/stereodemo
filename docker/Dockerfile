FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
#  ubuntu 22.04 doesn't work for some weird reason. glxgears crashes => no support for gui

ENV TZ=Europe/Prague
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:flexiondotorg/nvtop && apt-get install -y nvtop

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-is-python3 \
    mc \
    tmux \
    wget \
    curl \
    htop \
    git \
    git-lfs \
    # opengl tools
    mesa-utils \
    # tensorrt tools
    tensorrt-dev \
    python3-libnvinfer \
    # build tools
    cmake

# install github command line tool
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& apt update \
&& apt install gh -y


RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user

COPY docker/.bashrc .
COPY docker/.tmux.conf .


COPY setup.cfg .
COPY setup.py .

ENV PATH /home/user/.local/bin:$PATH
RUN pip3 install virtualenv && virtualenv -p python3 env

# activate environment in container
ENV PATH /home/user/env/bin:$PATH

RUN pip3 install .

RUN mkdir -p /home/user/workspace
WORKDIR /home/user/workspace

CMD ["/bin/bash"]