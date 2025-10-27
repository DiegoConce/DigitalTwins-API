FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126

# Copy project files
COPY src/ /app/src/
COPY templates/ /app/templates/
COPY data/ /app/data/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Create entrypoint script
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

echo "Starting API server..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 &
API_PID=\$!

echo "Waiting for services to be ready..."
sleep 3

echo "Populating data..."
python -m src.populate_data || echo "Data population failed or already populated"

echo "Services ready!"
wait \$API_PID
EOF

RUN chmod +x /app/entrypoint.sh

# Run the entrypoint script
CMD ["/app/entrypoint.sh"]
