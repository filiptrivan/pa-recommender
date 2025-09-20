FROM python:3.11-slim

# This is needed for implicit library
RUN apt-get update && apt-get install -y \
    libgomp1=14.2.0-19 \
    libomp-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

ENV OMP_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/lib

# Set the working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

CMD ["gunicorn", "-b", ":8080", "app:app"]