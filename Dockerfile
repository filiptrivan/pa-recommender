FROM python:3.11-slim

# This is needed for implicit library
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

# ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu

# Set the working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD exec gunicorn --bind :8080 --workers 1 --preload app:app