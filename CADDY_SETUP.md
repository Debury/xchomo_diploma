# Caddy Setup - Automatic HTTPS

Caddy je jednoduchý reverse proxy s automatickým SSL certifikátom cez Let's Encrypt.

## Konfigurácia

1. **Uprav Caddyfile** - zmeň doménu na svoju:
   ```
   climaterag.online {
       reverse_proxy web-api:8000
   }
   ```

2. **Nastav DNS záznamy**:
   - A záznam: `climaterag.online` → IP servera (159.65.207.173)
   - AAAA záznam (voliteľné): IPv6 adresa

3. **Spusti služby**:
   ```bash
   docker-compose up -d
   ```

## Čo Caddy robí automaticky:

- ✅ Automaticky získava SSL certifikát z Let's Encrypt
- ✅ Automaticky obnovuje certifikáty
- ✅ Presmeruje HTTP → HTTPS
- ✅ Reverse proxy na web-api:8000
- ✅ Presmeruje www → non-www (voliteľné)

## Porty

- **80**: HTTP (automaticky presmeruje na HTTPS)
- **443**: HTTPS (hlavný port)
- **8000**: FastAPI (len interné, cez Docker network)

## Prístup

Po nastavení DNS a spustení:
- `https://climaterag.online` → Vue.js frontend
- `https://climaterag.online/app/` → Vue.js SPA
- `https://climaterag.online/docs` → FastAPI dokumentácia
- `https://climaterag.online/rag/chat` → RAG endpoint

## Troubleshooting

**Caddy nezačína:**
```bash
docker-compose logs caddy
```

**SSL certifikát sa nevygeneruje:**
- Skontroluj, či DNS záznamy smerujú na správnu IP
- Počkaj na DNS propagáciu (5-15 minút)
- Skontroluj, či porty 80 a 443 sú otvorené

**Zmeniť doménu:**
1. Uprav Caddyfile
2. `docker-compose restart caddy`
