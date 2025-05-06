# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    cmake \
    python3-dev \
    libgl1-mesa-dev \
    libxrender1 \
    libxext6 \
    zlib1g-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Ensure no wrong 'fitz' package is present
RUN pip uninstall -y fitz || true

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pymupdf
# Copy the rest of the application code into the container
COPY . .

# Expose the port that the app will run on
EXPOSE 8080

# Command to run the application with Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "main:app"]
