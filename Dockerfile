# Use a standard Python base image.
# Using a "slim" image keeps the final container small.
FROM python:3.11-slim

# Install system dependencies with apt-get.
# `apt-get update` fetches the package list.
# `apt-get install -y` installs the packages non-interactively.
# `rm -rf /var/lib/apt/lists/*` cleans up the cache to reduce image size.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Set the working directory for your application.
WORKDIR /

# Copy your requirements file and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Set the command to run your application.
# Replace this with your actual application entrypoint.
CMD ["gunicorn", "-b", ":$PORT", "app:app"]
