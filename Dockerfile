FROM python:3.11-slim

WORKDIR /app

# Install unzip utility
RUN apt-get update && apt-get install -y unzip && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126

# Copy project files
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY data/ /app/data/

# Unzip datasets and models
RUN unzip data/data.zip -d data/ && rm data/data.zip

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod 755 /app/entrypoint.sh

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]
