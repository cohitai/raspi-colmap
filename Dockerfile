# Use the Python3.8 container image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Opencv dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Install docker
RUN apt-get -y install apt-transport-https ca-certificates curl gnupg2 software-properties-common
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"

#Create local data directory
RUN mkdir data

# Install the dependecies
RUN pip install -r requirements.txt

#Entry
CMD python main.py