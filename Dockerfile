# syntax=docker/dockerfile:1

# Build the wheel from the source tree.
FROM python:3.12-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src
RUN python -m build --wheel --outdir /wheels

# Runtime: the package with the HTTP server extra, run as a non-root user.
FROM python:3.12-slim AS runtime
# Add more extras at build time, for example:
#   docker build --build-arg EXTRAS=server,pdf,docx,sentence-transformers .
ARG EXTRAS=server,pdf,docx
LABEL org.opencontainers.image.title="persian-rag" \
      org.opencontainers.image.description="Persian-first retrieval-augmented generation toolkit with an HTTP API" \
      org.opencontainers.image.source="https://github.com/sadrmanesh/persian-rag" \
      org.opencontainers.image.url="https://sadrmanesh.github.io/persian-rag/" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Naser Sadrmanesh <info.sadrmanesh@gmail.com>"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PERSIAN_RAG_PORT=8000
COPY --from=build /wheels /tmp/wheels
RUN wheel="$(ls /tmp/wheels/*.whl)" \
    && pip install "${wheel}[${EXTRAS}]" \
    && rm -rf /tmp/wheels \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown app:app /data
USER app
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PERSIAN_RAG_PORT', '8000') + '/health', timeout=4)" || exit 1
ENTRYPOINT ["persian-rag"]
CMD ["serve", "--index", "/data/index", "--host", "0.0.0.0", "--port", "8000"]
