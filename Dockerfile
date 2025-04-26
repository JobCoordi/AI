FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

# 핵심: 모듈 인식 안 되는 문제 방지
ENV PYTHONPATH=/app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
