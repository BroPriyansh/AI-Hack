FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and prepare user
RUN pip install --upgrade pip
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install Python dependencies from setup.py and requirements.txt
RUN pip install --no-cache-dir -e .

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose the correct port for Render
EXPOSE 8080

# Start the Streamlit app
CMD ["streamlit", "run", "main.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]
