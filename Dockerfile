FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV RUN_XAI=1

EXPOSE 5000

CMD ["python", "app.py"]
