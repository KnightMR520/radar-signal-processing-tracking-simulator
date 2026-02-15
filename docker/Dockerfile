FROM python:3.11-slim

WORKDIR /app

# system deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# install python deps first for caching
COPY pyproject.toml poetry.lock* requirements.txt* /app/

# If you use requirements.txt:
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# If you use pyproject/poetry, uncomment this block and remove the requirements block above:
# RUN pip install --no-cache-dir poetry
# RUN if [ -f pyproject.toml ]; then poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi; fi

COPY . /app

# good defaults for clean logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# default command: run unit tests
CMD ["python", "-m", "pytest", "-q"]
