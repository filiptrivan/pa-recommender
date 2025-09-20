# Use a standard Python base image.
FROM python:3.11

# Install system dependencies with apt-get.
# `apt-get update` fetches the package list.
# `apt-get install -y` installs the packages non-interactively.
# `rm -rf /var/lib/apt/lists/*` cleans up the cache to reduce image size.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:$LD_LIBRARY_PATH

# Set the working directory for your application.
WORKDIR /app

# Copy your requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Set the command to run your application.
# Replace this with your actual application entrypoint.
CMD ["gunicorn", "-b", ":$PORT", "app:app"]
