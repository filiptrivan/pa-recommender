# FROM python:3.11

# # This is needed for implicit library
# RUN apt-get update && apt-get install -y \
#     libgomp1=14.2.0-19 \
#     libomp-dev \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# ENV OMP_NUM_THREADS=1
# ENV PYTHONUNBUFFERED=1
# ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

# # Set the working directory
# WORKDIR /app

# # Copy requirements file and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the rest of the application code
# COPY . .

# CMD ["gunicorn", "-b", ":8080", "app:app"]
FROM ubuntu:22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgomp1 \
    libomp-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python commands
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir numpy cython
RUN pip install --no-cache-dir --no-binary=implicit implicit
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 --timeout 0 app:app