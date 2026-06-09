.PHONY: install clean test docs pop

install:
	python -m pip install -e .

clean:
	rm -rf build/ curvanato.egg-info/ dist/ .pytest_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f ex_control.png ex_disease.png boxplot.png population_report.html

test:
	python -m pytest -v -s --cov=curvanato --cov-report=term-missing tests/

docs:
	@echo "Documentation generation not fully configured."
	@echo "If using Sphinx or MkDocs in the future, add the build command here."

pop:
	@echo "Running the synthetic population study..."
	python3 examples/caudate_report.py
