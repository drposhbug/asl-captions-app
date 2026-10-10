# Cloud hub (DigitalOcean App Platform): HUD mirror, events, alerts, flight feed. No mic, no GPU.
FROM python:3.12-slim
WORKDIR /app
COPY server/hub.py server/hud.html server/flight.py ./
ENV PORT=8080
EXPOSE 8080
CMD ["python", "hub.py"]
