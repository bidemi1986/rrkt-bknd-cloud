# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 8080

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "main:app"]
