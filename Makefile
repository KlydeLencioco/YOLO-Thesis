.PHONY: clean clean-test clean-pyc clean-build

PACKAGE = src

clean: clean-pyc clean-test

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr test-reports/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

verify: format lint

format-check:
	poetry run isort --check ${PACKAGE}
	poetry run black --check ${PACKAGE}

format:
	poetry run isort ${PACKAGE}
	poetry run black ${PACKAGE}

lint:
	poetry run flake8 ${PACKAGE}
