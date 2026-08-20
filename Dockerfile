FROM python:3.11-slim

WORKDIR /code

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY src/ src/
COPY app/ app/
COPY checkpoints/resnet18_frozen.pth checkpoints/resnet18_frozen.pth

# Hugging Face Spaces (Docker SDK) expects the app to listen on 7860.
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
