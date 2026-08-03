FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY shared/ ./shared/
RUN pip install --no-cache-dir -r requirements.txt && pip install -e shared/

COPY fl/ ./fl/
COPY server/ ./server/

EXPOSE 8080

CMD ["python", "-m", "fl.server.main"]
