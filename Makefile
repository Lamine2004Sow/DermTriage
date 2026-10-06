PYTHON ?= python3
VENV ?= .venv
VPY := $(VENV)/bin/python
PIP := $(VPY) -m pip

.PHONY: install splits preprocess test baselines features aggregate time-epoch clean

install: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

splits:
	$(VPY) scripts/make_splits.py

preprocess:
	$(VPY) scripts/preprocess.py

test:
	$(VPY) -m pytest

baselines:
	$(VPY) scripts/baselines.py

features:
	$(VPY) -m scripts.extract_features

aggregate:
	$(VPY) -m scripts.aggregate

time-epoch:
	$(VPY) scripts/time_epoch.py --max-batches 20

clean:
	rm -rf $(VENV)
