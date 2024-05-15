# You can adapt the base image based on the CUDA version installed on the device
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

#Set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install -y git
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

# Copy our application code
WORKDIR /restful-llama-3
COPY ./requirements.txt /restful-llama-3
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

EXPOSE 5000
CMD ["./start_app.sh"]