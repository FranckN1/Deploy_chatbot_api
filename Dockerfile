FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

# Installer les dépendances nécessaires pour compiler les paquets
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/

RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
