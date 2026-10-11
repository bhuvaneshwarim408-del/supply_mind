FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 SUPPLYMIND_PUBLIC=1
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY . .
EXPOSE 8000
# One worker: the what-if models and the rate limiter live in memory. Several threads keep pages responsive.
CMD ["sh", "-c", "gunicorn 'web:create_app()' --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 120"]
