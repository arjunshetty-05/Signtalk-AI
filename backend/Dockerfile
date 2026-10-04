# SignTalk AI — Backend Dockerfile (Person C, Prompt C3)
FROM python:3.10-slim

WORKDIR /app

# System deps for opencv / audio libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Socket.IO-wrapped ASGI app is the real entrypoint (wraps the FastAPI app)
CMD ["uvicorn", "api.socket_manager:socket_app", "--host", "0.0.0.0", "--port", "8000"]
