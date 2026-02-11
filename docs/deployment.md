# Deployment Guide (Production Web)

This guide deploys ContinualCode Web as a real website using Docker.

## 1. Prerequisites

- Docker + Docker Compose installed
- A valid `TINKER_API_KEY` (must start with `tml-`)
- DNS/domain pointing at your server (optional, for public HTTPS)

## 2. Build + Run

From repo root:

```bash
export TINKER_API_KEY=tml-...
export MODEL_NAME=moonshotai/Kimi-K2.5
export PORT=8765

docker compose up -d --build
```

Health check:

```bash
curl http://127.0.0.1:${PORT}/healthz
```

Open app:

```text
http://<server-ip>:8765
```

## 3. Reverse Proxy + HTTPS (Recommended)

Run Nginx (or Caddy/Traefik) in front of the container and terminate TLS there.

Minimal Nginx site config:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Then issue certs with certbot:

```bash
sudo certbot --nginx -d your-domain.com
```

## 4. Updates

```bash
git pull
docker compose up -d --build
```

## 5. Runtime Notes

- Sessions are lightweight conversation threads.
- The app uses one shared Tinker training/sampling runtime per server process.
- To switch to a different base model/checkpoint globally, restart with a different `MODEL_NAME`.
