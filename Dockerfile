# FROM python:3
FROM ubuntu:16.04

# install required binaries
RUN apt-get update \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y software-properties-common python-software-properties \
    && apt-get install graphviz -y \
    && apt-get install screen -y \
    && apt-get install htop -y \
    && apt-get install dieharder -y \
    && apt-get install dos2unix -y \
    && apt-get install openssh-client -y

# install pip requirements
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# copy project to docker image and convert this script to UNIX format
ADD src /adversarial-csprng/src
ADD get_results.sh /adversarial-csprng/
RUN dos2unix /adversarial-csprng/get_results.sh

# add missing directories
RUN cd adversarial-csprng