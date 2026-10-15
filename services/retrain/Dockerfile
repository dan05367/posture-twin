# 建置時的 context 是專案根目錄（需要把 ml/ 一起放進映像檔）
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /srv

COPY services/retrain/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml ./ml
COPY services/retrain/app ./app

RUN useradd -m appuser && mkdir /state && chown appuser /state
USER appuser

EXPOSE 8002
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health', timeout=2)" || exit 1

CMD ["python", "-m", "app.main"]
