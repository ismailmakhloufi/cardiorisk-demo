FROM python:3.11-slim

WORKDIR /app

# Node for frontend build
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

COPY requirements.txt* ./
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || pip install fastapi uvicorn pandas openpyxl python-multipart

COPY . .

# Build frontend
WORKDIR /app/frontend
RUN npm install && npm run build

WORKDIR /app

# Data dirs
RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
