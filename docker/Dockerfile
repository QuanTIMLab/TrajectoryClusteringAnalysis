FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python -m venv "$VIRTUAL_ENV"

WORKDIR /build
COPY requirements-app.txt .
RUN pip install setuptools wheel
RUN pip install -r requirements-app.txt
COPY setup.py README.md requirements.txt ./
COPY src ./src
RUN pip install --no-build-isolation --no-deps .

FROM python:3.11-slim AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHERRUSAGESTATS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 streamlit \
    && useradd --system --uid 10001 --gid streamlit --create-home streamlit \
    && mkdir -p /app /home/streamlit/.streamlit \
    && chown -R streamlit:streamlit /app /home/streamlit

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=streamlit:streamlit app ./app
COPY --chown=streamlit:streamlit data ./data
COPY --chown=streamlit:streamlit .streamlit ./.streamlit
USER 10001:10001
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app/streamlit_app.py"]
