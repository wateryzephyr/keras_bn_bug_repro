FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        ssh-client \
        sudo \
        vim \
        wget \
        python-dev \
        python-all-dev \
        python-pip \
        python-psutil \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install keras==2.2.4 tensorflow-gpu==1.13.1

ARG USER_NAME=developer
ARG USER_ID=1000
ARG GROUP_NAME=developer
ARG GROUP_ID=1000

RUN groupadd -g $USER_ID $USER_NAME
RUN useradd -s /bin/bash -m -u $USER_ID -g $GROUP_ID $USER_NAME
RUN chmod o+w /home/$USER_NAME
RUN echo $USER_NAME:$USER_NAME | chpasswd
RUN adduser $USER_NAME sudo

ENV HOME="/home/$USER_NAME"
ENV TERM=xterm

CMD ["/bin/bash"]
