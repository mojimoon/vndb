# Deployment

## 1. Cloudflare (one-time)

Requires Node.js 22+.

```bash
cd web
npm install
npx wrangler login
npx wrangler d1 create vndb          # put the printed database_id into web/wrangler.jsonc
npm run deploy                       # deploys the Worker and the frontend; tables come from the first imported snapshot
```

Custom domain: uncomment `routes` in `web/wrangler.jsonc` and fill in the domain (it must already be on Cloudflare), or add it in the dashboard under Worker → Settings → Domains & Routes. The Workers edge cache (Cache API) only works on a custom domain; on `*.workers.dev` every request reads the database.

## 2. Daily refresh (GitHub Actions)

Create a Cloudflare API token: avatar (top right) → **My Profile → API Tokens → Create Token**, use the **Edit Cloudflare Workers** template, then **+ Add more** for the permission **Account · D1 · Edit**. Under Account Resources pick your account; under Zone Resources pick your domain (or All zones).

Add these under the GitHub repository's **Settings → Secrets and variables → Actions**:

| Secret | Value |
| --- | --- |
| `CLOUDFLARE_API_TOKEN` | the token created above |
| `CLOUDFLARE_ACCOUNT_ID` | the Account ID on the right of the dashboard's Workers & Pages page |
| `D1_DATABASE_ID` | the database_id from `npx wrangler d1 list` |

Then:

- `refresh.yml` runs daily at 12:07 UTC (VNDB publishes the dump around 08:00 UTC): it downloads the dump, recomputes everything and imports it into D1. It can also be run by hand from the Actions page. Scheduled runs only trigger on the default branch.
- `deploy.yml` deploys whenever `web/` changes on `main`.
- `ci.yml` runs the tests; `dump-probe.yml` checks the dump's table layout.

## 3. Local development

Frontend, with the bundled **synthetic** sample data (not real VNDB data):

```bash
cd web
npm install
npm run db:seed:local
npm run dev                          # http://localhost:5173
```

Pipeline (Python 3.11+; a full run needs a few GB of RAM):

```bash
curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db && tar -I zstd -xf db.tar.zst -C db && rm db.tar.zst

cd pipeline
pip install -r requirements.txt
python -m vndb_rank --dump ../db --out out          # ~10 min; --skip-rankit / --skip-users / --skip-notes skip steps
npx --prefix ../web wrangler d1 execute vndb --local --file out/snapshot.sql   # import into local D1
python -m pytest                                     # tests (synthetic data)
```
