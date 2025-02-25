# Use a lightweight Python base image
FROM python:3.10-slim

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Upgrade pip and install essential Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only the requirements file first (for caching)
COPY requirements.txt .

# Install required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script
COPY main1.py .

# Expose the service port
EXPOSE 11434

# Default command to run your main script
CMD ["streamlit", "run", "main1.py"]