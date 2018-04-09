FROM python:3

# install required binaries
CMD pwd
RUN apt-get update \
    && apt-get install graphviz -y \
    && apt-get install screen -y \
    && apt-get install htop -y

# install pip requirements
ADD requirements.txt /
RUN pip3 install -r requirements.txt

# copy project to docker image
ADD src /adversarial-csprng/src

# add missing directories
RUN cd adversarial-csprng