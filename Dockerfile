FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir -U pip setuptools wheel

# Copy metadata + source (required because package-dir=src)
COPY pyproject.toml README.md /app/
COPY src /app/src

# Install the package (and deps declared in [project])
# If you keep imbalanced-learn under optional dev extras, use: ".[dev]"
RUN pip install --no-cache-dir .

# Now copy the rest (apps/, artifacts/, etc.)
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]