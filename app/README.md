# Cannabis maturity app

Mobile + API + GPU inference for cannabis flower maturity assessment (trichome + stigma analysis).

## Layout

| Directory | Purpose |
|-----------|---------|
| `api/` | FastAPI backend (analysis endpoint, Supabase storage + DB, optional Modal client) |
| `backend/` | ML code (detection, classification, maturity assessment) — used by API locally or by Modal |
| `modal/` | Modal app: GPU inference in the cloud; upload weights and deploy from here |
| `mobile/` | Expo (React Native) app |
| `supabase/` | Supabase config and migrations (analyses table, local dev) |
| `docs/` | Plans, architecture, and setup guides |

## Get running on your machine

**Start here:** [docs/SETUP.md](docs/SETUP.md) — clone, env, Supabase, Modal, run API.

More detail on Modal and Supabase: [docs/working-with-modal-and-supabase.md](docs/working-with-modal-and-supabase.md).

## Quick commands

- **API:** `cd api && doppler run -- uv run uvicorn main:app --reload --app-dir src`
- **Modal deploy:** `cd modal && uv run modal deploy inference.py`
- **Upload weights:** `cd modal && uv run modal run upload_weights.py`
- **Local Supabase:** `cd supabase && supabase start`

Environment: copy `api/.env.example` to `api/.env` and set `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (see SETUP.md).
