# adapt cuda version based on your cuda-version, TERMINAL: `nvidia-smi`
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Set up environment
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

# Create cache directory and set permissions
RUN mkdir -p /restful-llama-3/cache && chmod -R 777 /restful-llama-3/cache
# Make start_app.sh executable
RUN chmod +x /restful-llama-3/start_app.sh

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HF_HOME=/restful-llama-3/cache
ENV HF_DATASETS_CACHE=/restful-llama-3/cache

EXPOSE 7860
CMD ["./start_app.sh"]