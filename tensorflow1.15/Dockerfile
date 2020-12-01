FROM ubuntu

ENV TZ 'Europe/Berlin'
RUN echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y tzdata && \
    rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean

RUN apt-get update
RUN apt-get install -y software-properties-common 
RUN apt-get install -y wget 
RUN apt-get install -y git
RUN add-apt-repository ppa:deadsnakes/ppa -y && apt-get update
RUN apt-get install -y python3.6 python3.6-dev
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py
RUN rm -f /usr/local/bin/pip3
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 && ln -s /usr/local/bin/pip /usr/local/bin/pip3
RUN apt-get install -y cmake libopenmpi-dev zlib1g-dev
RUN apt-get install -y libglu1-mesa libglib2.0-0
RUN pip3 install tensorflow==1.15.0
RUN pip3 install git+https://github.com/hill-a/stable-baselines
RUN pip3 install Box2D

