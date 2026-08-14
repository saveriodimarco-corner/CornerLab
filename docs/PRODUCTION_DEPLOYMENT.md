# CornerLab Production Deployment Guide

This document is for the frozen Serie A production baseline tagged as `cornerlab-serie-a-v1`.

## Scope

This deployment is infrastructure and mobile operations only. It does not modify protected production logic:

- Serie A over_9_5 model
- Serie A over_10_5 model
- model artifact hashes
- DecisionEngine
- betting thresholds
- staking logic
- supported-market rules
- paper-trading scientific logic
- TOP / BUONA / MARGINALE semantics
- Premier League model status

The deployment package is prepared locally and is ready for a VPS provisioning step. No public server IP, domain, or credentials are assumed.

## 1. Provision server

1. Provision a small Linux VPS equivalent to Hetzner CX23 or similar.
2. Use Ubuntu 22.04 LTS or Debian 12.
3. Create a non-root user `cornerlab` with sudo access.
4. Enable SSH key authentication only.
5. Configure UFW firewall:
   - allow 22/tcp from admin IP only
   - allow 80/tcp if certbot HTTP verification is required
   - allow 443/tcp for HTTPS
   - deny all other inbound access
6. Install Docker and Docker Compose if using the packaged architecture below.
7. Ensure the host has persistent disk for `/var/lib/cornerlab`, `/srv/cornerlab`, and backups.

## 2. Clone repository

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates
sudo useradd -m -s /bin/bash cornerlab
sudo usermod -aG sudo cornerlab
su - cornerlab
git clone https://github.com/<your-org>/CornerLab.git /srv/cornerlab
cd /srv/cornerlab
git checkout cornerlab-serie-a-v1
```

## 3. Checkout production tag / deployment branch

Use the protected production tag on the deployment host:

```bash
git fetch --tags --all
git checkout cornerlab-serie-a-v1
```

If the repository uses a deployment branch in addition to the tag, ensure the branch points exactly to the same commit.

## 4. Configure environment variables

Create a local environment file that is never committed:

```bash
cp .env.example .env
chmod 600 .env
```

Required variables:

```dotenv
CORNERLAB_APP_PASSWORD=change-me
THE_ODDS_API_KEY=replace-me
API_FOOTBALL_KEY=replace-me
```

Other credential variables already used by the project should be added here if present in the environment.

Do not print secret values in logs. Keep `.env` ignored by Git. The repository already has `.env` in `.gitignore`.

## 5. Start CornerLab

Use Docker Compose for the simplest reliable production option.

The production image installs Python dependencies during `docker build`; the
application container does not install dependencies at startup. Runtime state
remains host-mounted in `data/`, `models/`, and `reports/`, including the
collector database, paper-trading history, settlement outputs, production
baseline manifest, and performance reports. Streamlit is internal to the Docker
network on port 8501; only Nginx publishes ports 80 and 443.

Before the first container start, make the writable persistent directories
owned by the non-root container UID:

```bash
sudo chown -R 10001:10001 /srv/cornerlab/data /srv/cornerlab/reports
```

```bash
cd /srv/cornerlab
docker compose -f docker-compose.production.yml up -d --build
```

If using the systemd alternative instead of Docker:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui/app.py --server.address 0.0.0.0 --server.port 8501
```

## 6. Enable startup on reboot

With Docker:

```bash
sudo systemctl enable docker
```

If using a systemd service, install the service file and enable it:

```bash
sudo cp deploy/cornerlab.service /etc/systemd/system/cornerlab.service
sudo systemctl daemon-reload
sudo systemctl enable cornerlab
sudo systemctl start cornerlab
```

## 7. Verify HTTPS

Place a reverse proxy in front of Streamlit and terminate TLS there.

Recommended:

- Nginx or Caddy as reverse proxy
- ACME/Let’s Encrypt certificate
- public HTTPS only on port 443
- Streamlit stays bound to localhost or Docker internal network, not public port 8501

Example reverse-proxy target:

```nginx
server {
    listen 443 ssl;
    server_name cornerlabpro.com;

    ssl_certificate /etc/letsencrypt/live/cornerlabpro.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cornerlabpro.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
    }
}
```

Ensure the app is reachable only via HTTPS and password-protected login, never by an open public Streamlit port.

## 8. Verify iPhone login

On the iPhone, open Safari and visit the HTTPS URL.

- Log in with the configured `CORNERLAB_APP_PASSWORD`
- Confirm the login flow loads
- Confirm the app presents: Dashboard, Storico, Performance, and the system status area
- Confirm the UI is readable on mobile and the app can be added to the Home Screen

Expected competition states:

- Serie A: OPERATIVO
- Premier League: IN PREPARAZIONE
- No EPL PLAY

## 9. Run manual prematch

Run the same production workflow as the local script:

```bash
python3 scripts/run_prematch.py
```

This must remain the single manual production action for prematch refresh.

## 10. Verify automatic scheduler

The optional scheduler uses the production wrappers and should run only during a
conservative configured match-day window. The wrappers hold a per-job OS lock,
skip duplicate state safely, append `data/operations/job_history.jsonl`, and
update `reports/operations_status.json`. They reuse the canonical prematch and
settlement workflows; they do not retrain or alter decision logic.

Install the provided systemd examples on the VPS:

```bash
sudo install -d -m 0750 /etc/cornerlab
sudo cp deploy/systemd/automation.env.example /etc/cornerlab/automation.env
sudo chmod 0640 /etc/cornerlab/automation.env
sudo cp deploy/systemd/cornerlab-*.service deploy/systemd/cornerlab-*.timer /etc/systemd/system/
sudo usermod -aG docker cornerlab
sudo systemctl daemon-reload
sudo systemctl enable --now cornerlab-prematch.timer cornerlab-settlement.timer
systemctl list-timers 'cornerlab-*'
```

Set `CORNERLAB_JOB_WINDOW_START` and `CORNERLAB_JOB_WINDOW_END` in UTC in
`/etc/cornerlab/automation.env`; do not add API credentials to this file.

Example schedule:

- one broad refresh in the morning
- one refresh approximately 60–90 minutes before kickoffs

Use a cron or systemd timer with a config file. Keep the schedule simple and configurable.

Example cron abstraction:

```bash
0 8 * * * /srv/cornerlab/.venv/bin/python3 /srv/cornerlab/scripts/run_prematch.py >> /var/log/cornerlab/prematch.log 2>&1
30 17 * * * /srv/cornerlab/.venv/bin/python3 /srv/cornerlab/scripts/run_prematch.py >> /var/log/cornerlab/prematch.log 2>&1
```

Make sure the manual button remains available in the app.

## 11. Verify backups

Daily local rotating backups are required before any external cloud backup is configured.

Recommended local backup layout:

```bash
/var/backups/cornerlab/daily/
/var/backups/cornerlab/weekly/
```

Minimum backup set:

- collector database
- paper-trading and settlement data
- run history
- production baseline manifest
- performance reports

Example local backup command:

```bash
cp -a /srv/cornerlab/data /var/backups/cornerlab/daily/$(date +%F)/
cp -a /srv/cornerlab/reports /var/backups/cornerlab/daily/$(date +%F)/
cp -a /srv/cornerlab/models /var/backups/cornerlab/daily/$(date +%F)/
```

Optional cloud integration to Google Drive or Dropbox can be added later without making it a runtime dependency.

Retention: at least 14 daily backups.

## 12. Update application safely later

1. Stop the service or pull the new tag in a maintenance window.
2. Verify the target commit is tagged as the new production release.
3. Re-run the prematch workflow and sanity checks.
4. Confirm model hashes are unchanged for the frozen baseline.
5. Restart the app and validate the mobile login and dashboard.

## Rollback instructions

If a production update fails:

1. Stop the service.
2. Revert to the previous git tag or commit.
3. Restore the previous persistent runtime data from backup.
4. Restart the app.
5. Re-run the health checks and prematch workflow.
6. Confirm the app returns to the known-good baseline.

## Deployment readiness note

This repository is prepared for a VPS installation, but the live remote server, domain, and credentials are not supplied in this workspace. Therefore, the deployment is in a local provisioning-ready state only.
