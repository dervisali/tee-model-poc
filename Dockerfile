# Use the official Python lightweight image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Cloud Run injects the $PORT environment variable (defaults to 8080)
# Streamlit needs to be told to listen on 0.0.0.0 and on this port
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
