FROM python:3.8.12-buster

RUN apt-get -y update && apt-get install -y \
    make \
    gcc \
    git \
    vim

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /workspace
WORKDIR /workspace
