FROM python:3.11-slim

WORKDIR /code

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt huggingface_hub

COPY src/ src/
COPY app/ app/

# Checkpoint is hosted on the Hugging Face Hub (public model repo) rather than
# committed to git, since it's a 44MB binary that doesn't belong in source
# control history. Downloaded at build time so the running container has no
# runtime dependency on an external service.
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='abir171/pneumonia-xray-resnet18', filename='resnet18_frozen.pth', local_dir='checkpoints')"

# Render injects $PORT at runtime; default to 7860 for local/other hosts
# (e.g. Hugging Face Spaces' Docker SDK, which expects 7860 specifically).
EXPOSE 7860
ENV PORT=7860
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
