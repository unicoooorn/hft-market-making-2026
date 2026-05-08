# Makefile for Python linting, type checking, and formatting
# Requires: python3, pip, virtualenv (optional)
#
# Targets:
#   setup         - create virtual environment and install dependencies
#   check         - run Ruff linter + Pyre type checker
#   lint          - alias for 'ruff check'
#   format        - format code with Ruff
#   typecheck     - run Pyre type checker
#   profile       - run profiler and show results
#   bench         - run benchmark
#   clean         - remove virtual environment and cache files
#   help          - show this help

# Tool commands (override if needed)
PYTHON      := python3
VENV        := .venv
RUFF_CMD    := ruff
# Replace 'pyre' with actual 'pyrefly' command if different
PYREFLY_CMD := pyrefly

# ---------------------------------------------------------------------
# Helper: ensure ruff and pyrefly are installed
# ---------------------------------------------------------------------
.venv/bin/ruff:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install ruff
ifneq ($(PYREFLY_CMD), pyre)
	# Install custom pyrefly – adjust command as needed
	$(VENV)/bin/pip install $(PYREFLY_CMD)
else
	$(VENV)/bin/pip install pyre-check
endif

# ---------------------------------------------------------------------
# Setup target (creates venv and installs tools)
# ---------------------------------------------------------------------
.PHONY: setup
setup: $(VENV)/bin/ruff

# ---------------------------------------------------------------------
# Ruff linting (check only, no auto-fix)
# ---------------------------------------------------------------------
.PHONY: lint
lint: setup
	$(VENV)/bin/$(RUFF_CMD) check .

# ---------------------------------------------------------------------
# Ruff formatting (auto-fix style)
# ---------------------------------------------------------------------
.PHONY: format
format: setup
	$(VENV)/bin/$(RUFF_CMD) format .

# ---------------------------------------------------------------------
# Type checking with Pyre (or pyrefly)
# ---------------------------------------------------------------------
.PHONY: typecheck
typecheck: setup
	pyrefly check .

# ---------------------------------------------------------------------
# Full check (lint + type check)
# ---------------------------------------------------------------------
.PHONY: check
check: lint typecheck

# ---------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------
.PHONY: profile
profile: setup
	$(VENV)/bin/python scripts/profiler.py --sort cumulative --lines 30

.PHONY: profile-time
profile-time: setup
	$(VENV)/bin/python scripts/profiler.py --sort time --lines 30

.PHONY: profile-save
profile-save: setup
	$(VENV)/bin/python scripts/profiler.py --output profile.prof
	@echo "View with: snakeviz profile.prof"

# ---------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------
.PHONY: bench
bench: setup
	$(VENV)/bin/python scripts/benchmark_parquet_read.py data/cmf/lob.parquet 50000

.PHONY: bench-full
bench-full: setup
	$(VENV)/bin/python scripts/benchmark_parquet_read.py data/cmf/lob.parquet 500000

# ---------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# ---------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  setup       : create virtual env and install ruff + pyrefly"
	@echo "  lint        : run ruff linter"
	@echo "  format      : run ruff formatter"
	@echo "  typecheck   : run pyrefly type checker"
	@echo "  check       : lint + typecheck"
	@echo "  profile     : run profiler (sorted by cumulative time)"
	@echo "  profile-time: run profiler (sorted by internal time)"
	@echo "  profile-save: save profile data for snakeviz visualization"
	@echo "  bench       : run benchmark (50k rows)"
	@echo "  bench-full  : run benchmark (500k rows)"
	@echo "  clean       : remove venv and cache files"