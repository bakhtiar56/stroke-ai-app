FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

COPY src/ /app/src/
COPY apps/ui/ /app/apps/ui/

EXPOSE 8501

CMD ["streamlit", "run", "apps/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]