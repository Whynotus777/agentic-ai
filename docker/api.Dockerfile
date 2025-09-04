FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY common/ ./common/

EXPOSE 8000

CMD ["python", "-m", "api.gateway"]
