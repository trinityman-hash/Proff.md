# DRLE Cloudflare Workers Free deployment

This deployment serves the existing seed model as a static asset and runs a small deterministic inference runtime in a Cloudflare Worker. It is a test deployment, not a general-purpose LLM service. No account token or secret is stored in this repository.

## Requirements
- Node.js 20.0+ (Node 22 LTS recommended)
- A Cloudflare account with Workers enabled
- Wrangler CLI (installed locally through the project dependency)

## Install and test locally

```bash
npm install
npm run test:worker
npm run dev
```

Wrangler serves the Worker and `public/model.json` through the `ASSETS` binding. Test:

```bash
curl http://localhost:8787/health
curl -X POST http://localhost:8787/predict -H 'content-type: application/json' -d '{"context":"a deterministic engine","top_k":5}'
curl -X POST http://localhost:8787/generate -H 'content-type: application/json' -d '{"prompt":"a deterministic engine","max_tokens":12}'
```

## Deploy to Workers

```bash
npx wrangler login
npm run deploy
```

Wrangler will display the deployed `workers.dev` URL. The configuration publishes the `public/` directory as static assets and binds it as `ASSETS`; the Worker reads `/model.json` from that binding. Confirm the deployed `/health`, `/predict` and `/generate` routes after deployment.

## API

- `GET /health` — service status.
- `POST /predict` — JSON `{ "context": "...", "top_k": 8, "min_count": 1, "backoff": true }`.
- `POST /generate` — JSON `{ "prompt": "...", "max_tokens": 24 }`.

Limits enforced by this application: 16 KiB declared request body, 4,000 input characters, `top_k` 1–20, and generation 0–64 tokens. Cloudflare platform limits also apply. This prototype does not include authentication, abuse protection, per-user quotas or private data handling. Keep it to trusted testers until those are implemented. CORS is permissive for prototype convenience and should be restricted to the production frontend origin before wider use.

## Important runtime limitation

The seed model is approximately 257 KB as JSON and is loaded into isolate memory on first use. The current Worker is intentionally simple and scans candidates for each matching context order. It is suitable for validating deployment and API behavior, not for claiming high throughput. Measure CPU time and memory in Cloudflare's actual environment before expanding the model. A larger model should use a compact binary representation and an access pattern designed around the Worker CPU and subrequest budgets.
