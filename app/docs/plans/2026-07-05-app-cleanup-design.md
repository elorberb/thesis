# App cleanup — safe organization (2026-07-05)

## Goal

Make `app/` easy to understand at a glance: one source of truth per module, no dead
or duplicated code, no stray files. Behavior must not change and tests must stay green.

## Scope

Only `app/`. The research code at the repo root (`src/`, `notebooks/`, `scripts/`,
root `supabase/`) is intentionally left untouched.

## Problem

The app carried two parallel copies of several modules, which makes the codebase
confusing for a newcomer:

- `app/backend/src/` had a legacy flat layout (`trichome_detector.py`, `stigma_detector.py`,
  `models.py`, `color_classifier.py`, `crop_extractor.py`, `maturity_assessor.py`,
  `annotation_renderer.py`) that was fully superseded by the `cannabis_maturity/`
  package. Only `cannabis_maturity/` is packaged (`pyproject.toml`) and imported by the
  API, Modal, and every test.
- `app/api/src/services/` had two unused service classes (`database.py`'s
  `SupabaseDatabaseService` and `storage.py`'s `SupabaseStorageService`); the app wires
  up `DatabaseService` (`database_service.py`) and `StorageService` (`storage_service.py`)
  instead.
- `app/mobile/` contained an editor temp artifact (`tsconfig.json.<random-suffix>`).

## Changes

Deleted (behavior-preserving — none were imported anywhere in `app/`):

- `app/api/src/services/database.py`
- `app/api/src/services/storage.py`
- `app/backend/src/annotation_renderer.py`
- `app/backend/src/color_classifier.py`
- `app/backend/src/crop_extractor.py`
- `app/backend/src/maturity_assessor.py`
- `app/backend/src/models.py`
- `app/backend/src/stigma_detector.py`
- `app/backend/src/trichome_detector.py`
- `app/mobile/tsconfig.json.<random-suffix>` (stray temp file)

## Result

- `app/backend/src/` is now just the `cannabis_maturity/` package.
- `app/api/src/services/` contains only services that are actually used.

## Verification

- `app/backend`: `uv run pytest tests/` → 16 passed.
- No dangling imports of any deleted module remain in `app/` (grep clean).
- `app/api` tests could not run due to a pre-existing, unrelated issue: `src/auth.py`
  creates a Supabase client at import time from `settings.supabase_anon_key`, but
  `tests/conftest.py` only sets `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`. This predates
  the cleanup and is out of scope here.

## Out of scope / follow-ups

- Add `SUPABASE_ANON_KEY` to `app/api/tests/conftest.py` so the API suite can collect.
- Fix the import-sort (ruff `I001`) in `app/api/src/auth.py`.
