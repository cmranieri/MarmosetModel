FROM python:3.8.12-buster

RUN apt-get -y update && apt-get install -y \
    make \
    gcc \
    git \
    vim

RUN pip install neuron==8.0.2 &\
    pip install netpyne==1.0.1 &\
    pip install inspyred==1.0 &\
    pip install scikit-learn==1.0.2

RUN mkdir -p /workspace
WORKDIR /workspace
