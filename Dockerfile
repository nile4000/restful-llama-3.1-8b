# Use the official CUDA runtime base image, check it via nvidia-smi
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04

# Set up environment and install necessary packages in one RUN command to reduce the number of layers
RUN apt-get update && \
    apt-get install --no-install-recommends --no-install-suggests -y \
    curl \
    git \
    unzip \
    python3 \
    python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /restful-llama-3

# Copy requirements first to leverage Docker layer caching
COPY ./requirements.txt /restful-llama-3
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create model cache directory and set permissions
RUN mkdir -p /restful-llama-3/cache && chmod -R 777 /restful-llama-3/cache

# Make the start script executable
RUN chmod +x /restful-llama-3/start_app.sh

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HF_HOME=/restful-llama-3/cache
ENV HF_DATASETS_CACHE=/restful-llama-3/cache

# Expose HF port
EXPOSE 7860

# Specify the command to run the application
CMD ["./start_app.sh"]
