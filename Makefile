PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/python -m pip

.PHONY: install clean

install: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)

clean:
	rm -rf $(VENV)
