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

# Run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]