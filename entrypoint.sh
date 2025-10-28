#!/bin/bash
set -e

echo "Populating data..."
python -m src.populate_data || echo "Data population failed or already populated"

echo "Starting API server..."
uvicorn src.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

wait $API_PID