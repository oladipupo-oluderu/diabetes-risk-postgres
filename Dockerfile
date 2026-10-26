FROM python:3.11-slim
WORKDIR /app

# Install dependencies first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The database runs in a separate container (see docker-compose.yml).
# DATABASE_URL is provided by Compose at runtime.
CMD ["python", "src/pipeline.py"]
