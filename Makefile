PYTHON = .venv/bin/python
RUFF   = .venv/bin/ruff
PYWARN = -W ignore::UserWarning

export MPLBACKEND = Agg
export PYTHONPATH = src

MAIN = $(wildcard src/parkinsons/main.py)
ANALYSIS = $(wildcard src/parkinsons/analysis/*.py)
DEMOS = $(wildcard src/parkinsons/demos/*.py)
SCHEMAS = $(wildcard docs/schemas/*.dot)

.PHONY: help all clean ruff main analysis demos schemas report serve-api

help:
	@echo "Available targets:"
	@echo " make help          - Show available targets"
	@echo " make all           - Run complete pipeline (clean + ruff + main + analysis + demos + schemas + report)"
	@echo " make clean         - Remove generated results and report build artifacts"
	@echo " make ruff          - Format and auto-fix Python code with ruff"
	@echo " make main          - Run main pipeline"
	@echo " make analysis      - Run analysis scripts"
	@echo " make demos         - Run demo scripts"
	@echo " make schemas       - Compile docs/schemas/*.dot to tmp/*.png"
	@echo " make report        - Compile LaTeX report"
	@echo " make serve-api     - Start the FastAPI app with uvicorn"

all:
	@$(MAKE) --no-print-directory clean
	@$(MAKE) --no-print-directory ruff
	@$(MAKE) --no-print-directory main
	@$(MAKE) --no-print-directory analysis
	@$(MAKE) --no-print-directory demos
	@$(MAKE) --no-print-directory schemas
	@$(MAKE) --no-print-directory report

clean:
	@echo "Cleaning generated results..."
	@rm -rf results/figures/*
	@rm -rf results/tables/*
	@echo "Cleaning report build artifacts (keeping .tex and .bib)..."
	@find report/ -type f ! -name '*.tex' ! -name '*.bib' -delete
	@echo "Cleaning tmp/*.png..."
	@rm -rf tmp/

ruff:
	@echo "Running ruff: format + fix..."
	@$(RUFF) format src/parkinsons
	@$(RUFF) check --fix src/parkinsons || true

main:
	@echo "Running main pipeline..."
	@$(PYTHON) $(PYWARN) $(MAIN)

analysis:
	@echo "Running analysis scripts..."
	@for script in $(ANALYSIS); do \
		echo "Running $$script"; \
		$(PYTHON) $(PYWARN) $$script; \
	done

demos:
	@echo "Running demo scripts..."
	@for script in $(DEMOS); do \
		echo "Running $$script"; \
		$(PYTHON) $(PYWARN) $$script; \
	done

schemas:
	@echo "Compiling schema diagrams to PNG..."
	@mkdir -p tmp
	@for file in $(SCHEMAS); do \
		name=$$(basename $$file .dot); \
		echo "  $$file -> tmp/$$name.png"; \
		dot -Tpng $$file -o tmp/$$name.png; \
	done

report:
	@echo "Compiling LaTeX report..."
	@cd report && latexmk -pdf -interaction=nonstopmode main.tex

serve-api:
	@echo "Starting FastAPI app at http://127.0.0.1:8000 ..."
	@$(PYTHON) -m uvicorn parkinsons.api.app:app --host 127.0.0.1 --port 8000
