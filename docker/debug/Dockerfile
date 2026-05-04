FROM python:3.12-slim

ENV HOME=/sandbox/home \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt /tmp/
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-dev.txt

COPY docker/debug/entrypoint.sh /usr/local/bin/akashic-debug
RUN chmod +x /usr/local/bin/akashic-debug

ENTRYPOINT ["akashic-debug"]
CMD ["run"]
