PY=.venv/bin/python

.PHONY: setup data train test lint
setup:
	python3 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"

data:
	$(PY) scripts/download_data.py

train: data
	$(PY) -m modelguard.train

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

.PHONY: monitors
monitors: train
	$(PY) scripts/run_monitors.py

.PHONY: simulate
simulate: train
	$(PY) scripts/simulate.py

.PHONY: demo
demo: train
	$(PY) scripts/make_demo_db.py
	MODELGUARD_DB=data/demo.db .venv/bin/streamlit run src/modelguard/dashboard.py

.PHONY: eval
eval: train
	rm -rf results/eval results/tables
	$(PY) scripts/evaluate.py tune
	$(PY) scripts/evaluate.py report
	$(PY) scripts/make_tables.py
