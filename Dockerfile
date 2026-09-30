# One image, two Lambda handlers (API + worker). Container images allow up to 10 GB,
# which torch + sentence-transformers needs; zip packages cap at 250 MB.
FROM public.ecr.aws/lambda/python:3.12

# CPU-only torch keeps the image ~1 GB instead of ~5 GB.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY app ./app
COPY data ./data
RUN pip install --no-cache-dir ".[embeddings,observability]"

# Bake the embedding model into the image: no download at cold start, no Hugging Face dependency at runtime.
ENV EMBEDDING_CACHE_DIR=/opt/models HF_HUB_OFFLINE=1
RUN HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', cache_folder='/opt/models')"

CMD ["app.worker.handler"]
