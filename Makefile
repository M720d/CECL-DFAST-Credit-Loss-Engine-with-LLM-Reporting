.PHONY: all setup download convert panel pd hazard lgd cecl dfast llm test reports private clean-derived

all: convert panel pd hazard lgd cecl dfast test   ## phases 1-6 + tests; then `make llm` and `make reports`

setup:
	uv sync

download:            ## public inputs (Freddie Mac loan files are manual, see README)
	uv run python scripts/download_public.py

convert:             ## raw pipe-delimited -> typed parquet (idempotent)
	uv run python scripts/convert_freddie.py

panel:               ## Phase 1: outcomes, PD snapshots, QC, vintage tables
	uv run python scripts/build_panel.py

pd:                  ## Phase 2: scorecard, LightGBM challenger, recalibration test
	uv run python scripts/train_pd.py
	uv run python scripts/recalibrate_pd.py

hazard:              ## Phase 3: macro, state HPI betas, competing-risk hazard, backtest
	uv run python scripts/build_hazard_panel.py
	uv run python scripts/train_hazard.py

lgd:                 ## Phase 4: LGD (cure x severity) and EAD validation
	uv run python scripts/train_lgd.py

cecl:                ## Phase 5: CECL allowance, sensitivities, Jan-Jun 2026 backtest (~20 min)
	uv run python scripts/run_cecl.py

dfast:               ## Phase 6: 9-quarter DFAST, crisis replay, downturn LGD
	uv run python scripts/run_dfast.py

llm:                 ## Phase 7: RAG index, MDD drafting, traceability, eval (needs Ollama running)
	uv run python scripts/run_llm_reporting.py

test:
	uv run pytest -q

reports:             ## render README.md and the model development document (MD + PDF) from metrics
	uv run python scripts/build_reports.py

private:             ## interview guide PDF (gitignored)
	uv run python private/build_guide.py

clean-derived:       ## remove derived data, keep raw downloads
	rm -rf data/interim data/processed
