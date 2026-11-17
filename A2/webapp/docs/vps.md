# Run on a VPS shared with other projects

One Ubuntu server runs this app and a few other projects, each as its own Docker
Compose project. A single Cloudflare Tunnel brings HTTPS traffic in for all of them, so
the server opens no web ports and needs no certificates. This app sits behind HTTP
Basic Auth at its nginx gateway.

```text
browser ─HTTPS─▶ Cloudflare (DNS, TLS) ─tunnel─▶ cloudflared ──┐ Docker network "edge"
                                                                ├─▶ sleep-eeg-web (nginx, Basic Auth) ─▶ api, worker, db, qdrant
                                                                ├─▶ project-b-web ─▶ its own services
                                                                └─▶ project-c-web ─▶ its own services
```

- Each project's gateway joins the shared network `edge` under a unique alias. The tunnel
  sends each hostname to one alias, for example `eeg.example.com → http://sleep-eeg-web:80`.
- The API, worker, database and Qdrant of a project stay on that project's private
  network: the tunnel and other projects cannot reach them.
- No project publishes host ports. Docker-published ports bypass `ufw`, and two projects
  would collide on ports such as 5432.

The files for this app are [`compose.vps.yaml`](../compose.vps.yaml) and
[`.env.vps.example`](../.env.vps.example).

## 1. What you need

- **A domain on Cloudflare** (free plan), with its nameservers moved to Cloudflare. A
  named tunnel needs one; the Quick Tunnel of the [Mac demo](mac-demo.md) is not meant
  to stay up.
- **A VPS**: Ubuntu 24.04 LTS, x86_64.

This app's own footprint with the full 195,469-epoch library. The peak is the highest
value during the whole end-to-end suite, which uploads and runs both models on the
one-hour sample:

| | Idle | Peak | Cap in `compose.vps.yaml` |
| --- | --- | --- | --- |
| Qdrant | 0.9–1.2 GiB | 0.9 GiB (indexing included) | 2 GiB |
| api | 0.1–0.8 GiB | 0.75 GiB | 1.5 GiB, 2 CPUs |
| worker | 0.2–0.4 GiB | 0.45 GiB | 3 GiB, `TORCH_THREADS` CPUs |
| PostgreSQL | 0.03–0.1 GiB | 0.08 GiB | 1 GiB |
| nginx | 10 MiB | 11 MiB | 128 MiB |

The worker's cap leaves room for full nights, which are about eight times longer than
the sample. Disk: backend image 2 GB, library 4.1 GB, Qdrant volume 2.8 GB, plus uploads
and PostgreSQL. Caps are ceilings, not reservations: what has to fit in RAM is the sum of
what every project really uses (`docker stats`).

| Server | Fits |
| --- | --- |
| 4 vCPU, 8 GB RAM, 80 GB disk | this app plus 2–3 small web projects |
| 4–8 vCPU, 16 GB RAM, 160 GB disk | the other projects also run databases or ML models |

## 2. Prepare the server (once)

Log in as a sudo user with an SSH key, then:

```bash
# SSH: keys only, no root login.
sudo tee /etc/ssh/sshd_config.d/10-hardening.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
EOF
sudo systemctl reload ssh

# Firewall: SSH only. The tunnel connects out, so no web ports are opened.
sudo ufw allow OpenSSH && sudo ufw enable

# Security updates and 4 GB of swap.
sudo apt update && sudo apt install -y unattended-upgrades
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Install Docker Engine and the Compose plugin from Docker's apt repository
(<https://docs.docker.com/engine/install/ubuntu/>). Compose 2.24.4 or later is needed for
`!reset`. Then cap container logs, which otherwise grow until the disk is full, and
create the shared network:

```bash
sudo usermod -aG docker "$USER"      # log out and in again
sudo tee /etc/docker/daemon.json <<'EOF'
{ "log-driver": "json-file", "log-opts": { "max-size": "10m", "max-file": "3" } }
EOF
sudo systemctl restart docker

docker network create edge
sudo mkdir -p /srv && sudo chown "$USER" /srv
```

Every project goes under `/srv/<project>`.

## 3. Shared Cloudflare Tunnel (once)

1. In the Cloudflare dashboard, go to **Zero Trust → Networks → Tunnels → Create a
   tunnel**, choose **Cloudflared**, name it after the server, and copy the token from the
   install command shown (the long value after `--token`). Do not run that install
   command: the connector runs in Docker below.
2. On the server, create `/srv/edge/compose.yaml`:

   ```yaml
   name: edge
   services:
     cloudflared:
       image: cloudflare/cloudflared:2026.11.0   # pin a release; update deliberately
       command: tunnel --no-autoupdate run
       environment:
         TUNNEL_TOKEN: ${TUNNEL_TOKEN:?set TUNNEL_TOKEN in .env}
       networks: [edge]
       restart: unless-stopped
   networks:
     edge:
       external: true   # created once with `docker network create edge`
   ```

   Create `/srv/edge/.env` with `TUNNEL_TOKEN=<token>`, then `chmod 600 .env` and run
   `docker compose up -d` in `/srv/edge`. The tunnel shows as **Healthy** in the dashboard.
3. In the tunnel's **Public hostnames** (called **Published application routes** in newer
   dashboards), add `eeg.<your domain>` with service type **HTTP** and URL
   `sleep-eeg-web:80`. Cloudflare creates the DNS record.

Each later project adds one hostname here; DNS and the server stay unchanged.

## 4. Deploy this app

```bash
cd /srv
git clone <this repository> sleep-eeg-app     # a read-only deploy key works for a private repo
cd sleep-eeg-app/A2/webapp
cp .env.vps.example .env
sed -i "s/^POSTGRES_PASSWORD=$/POSTGRES_PASSWORD=$(openssl rand -hex 24)/" .env
chmod 600 .env
```

Set the Basic Auth login in `.env` now (see the security note in section 7).
`POSTGRES_PASSWORD` must be set before the first start: PostgreSQL reads it only when
its volume is created. `.env` also sets `COMPOSE_FILE` and `COMPOSE_PROFILES`, so a plain
`docker compose` command in this directory always uses `compose.vps.yaml` and starts
Qdrant. All commands below rely on that.

Checkpoints and the library are not in Git. Copy them from the Mac, from `A2/webapp`.
`-a` keeps modification times, which the API uses to notice a new library version.
`eval_neighbours.npz` is only used by the evaluation script.

```bash
rsync -a --progress backend/checkpoints/ <server>:/srv/sleep-eeg-app/A2/webapp/backend/checkpoints/
rsync -a --progress --exclude eval_neighbours.npz \
  backend/library/ <server>:/srv/sleep-eeg-app/A2/webapp/backend/library/
```

Start the stack, then fill Qdrant and the `library_epochs` table from the library files.
`scripts/` is not in the image, so it is mounted into a one-off container. The times
are from the rehearsal on a Mac:

```bash
docker compose up -d --build --wait

run_script() {
  docker compose run --rm -v ./backend/scripts:/app/scripts:ro api python scripts/build_library.py "$@"
}
run_script index      # about 2.5 minutes for 195,469 epochs
run_script load-db    # about 10 s
run_script verify     # "OK: files, Qdrant and PostgreSQL agree"
```

Check it from the server, then open `https://eeg.<your domain>`. The browser asks for the
Basic Auth login.

```bash
docker compose ps                                                       # all healthy
docker compose exec web wget -qO- http://127.0.0.1/api/health           # no login needed
docker compose exec web wget -qO- http://admin:admin@127.0.0.1/api/v1/library | head -c 200
```

## 5. Run it

| Task | Command (in `/srv/sleep-eeg-app/A2/webapp`) |
| --- | --- |
| Update | `git pull && docker compose up -d --build --wait` (migrations run when the API starts) |
| Logs | `docker compose logs -f --tail 100 api worker` |
| Resource use | `docker stats` |
| Change the login | edit `.env`, then `docker compose up -d web` |
| New library version | rsync it, then `run_script index --recreate`, `run_script load-db`, `run_script verify` |

**Backups.** PostgreSQL holds recordings' metadata and results. Uploaded files are in the
`sleep-webapp_appdata` volume and are deleted after `RETENTION_DAYS` anyway. Qdrant and
the library are rebuilt from the files on the Mac, so they need no backup. A nightly dump
from `crontab -e` (after `mkdir -p /srv/backups`), to copy off the server, for example
with `rclone`:

```cron
15 3 * * * cd /srv/sleep-eeg-app/A2/webapp && docker compose exec -T db pg_dump -U sleep -Fc sleep > /srv/backups/sleep-$(date +\%F).dump && find /srv/backups -name 'sleep-*.dump' -mtime +14 -delete
```

Restore with `docker compose exec -T db pg_restore -U sleep -d sleep --clean < <file>`.

**Uptime.** Point a monitor at `https://eeg.<your domain>/api/health`, the one path
without a login. It answers only when nginx, the API and PostgreSQL are all up.

**Reboots.** Every service has `restart: unless-stopped`, the tunnel included. Reboot
once after the first deploy (`sudo reboot`) and check that `docker compose ps` is healthy
again without any manual step.

## 6. Add another project

1. Give it its own directory `/srv/<project>`, Compose `name:`, `.env`, volumes and
   database container. Sharing one PostgreSQL between projects couples their upgrades
   and backups.
2. Publish no `ports:`. Use `!reset []` if its base file has some, as `compose.vps.yaml`
   does.
3. Attach only its gateway to `edge`, with an alias unique on the server, and add a
   tunnel hostname that points at it:

   ```yaml
   services:
     web:
       networks:
         default: {}
         edge: { aliases: [project-b-web] }
   networks:
     edge: { external: true }
   ```

   Never attach internal services. Names on `edge` resolve from every gateway, and this
   app's nginx proxies to `api`: an `api` service of another project on `edge` could
   receive its requests.
4. Set `deploy.resources.limits` on its services, then check with `docker stats` that
   real use across all projects fits the RAM.
5. Give it its own login: Basic Auth at its gateway, its own accounts, or Cloudflare
   Access.

## 7. Security notes

- **Basic Auth is one shared login.** Whoever has it can upload, export and delete every
  recording. `admin`/`admin` is the first pair bots try: change it in `.env` before
  sharing the URL. The password travels over HTTPS to Cloudflare and through the
  encrypted tunnel. For per-person logins, add a Cloudflare Access application in front
  of the hostname later; no code changes are needed. Access and Basic Auth can be stacked.
- `/api/health` is open on purpose. It reports only that the stack is up and how many
  model releases are installed.
- The web container refuses to start with only one of `BASIC_AUTH_USER` and
  `BASIC_AUTH_PASSWORD` set. Compose refuses to start the stack when either is empty, and
  likewise for `POSTGRES_PASSWORD`.
- Cloudflare's free plan limits a request body to 100 MB (this app accepts 64 MiB) and
  waits 100 s for a response. Inference runs as a background job, so no request comes
  close.

## Rehearse on the Mac

The same files run on Docker Desktop. A separate project name keeps the rehearsal
apart from the development stack and its volumes. A small extra file publishes the
gateway on localhost for Playwright:

```bash
docker network create edge
cp .env.vps.example /tmp/vps.env    # set POSTGRES_PASSWORD in it
printf 'services:\n  web:\n    ports: ["127.0.0.1:8090:80"]\n' > /tmp/rehearsal.yaml
vps() { docker compose -p sleep-vps-test --env-file /tmp/vps.env \
          -f compose.yaml -f compose.vps.yaml -f /tmp/rehearsal.yaml "$@"; }
vps up -d --build --wait
vps run --rm -v ./backend/scripts:/app/scripts:ro api python scripts/build_library.py index
# ... load-db, verify, as in section 4

# What cloudflared sees: 401 without the login, 200 with it.
docker run --rm --network edge curlimages/curl:8.11.1 -s -o /dev/null -w '%{http_code}\n' http://sleep-eeg-web/
(cd frontend && E2E_BASE_URL=http://127.0.0.1:8090 E2E_USER=admin E2E_PASSWORD=admin npx playwright test)

vps down -v && docker network rm edge
```

`--env-file` is used instead of the project's `.env`. The `-f` options take precedence
over its `COMPOSE_FILE`; its `COMPOSE_PROFILES` still starts Qdrant.
