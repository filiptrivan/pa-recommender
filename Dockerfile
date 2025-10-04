
# Use NVIDIA CUDA runtime base image (includes GPU libraries)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" points to python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

ENV OMP_NUM_THREADS=1

# Set the working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

CMD ["gunicorn", "-b", ":8080", "app:app", "--workers", "1", "--timeout", "3600"]