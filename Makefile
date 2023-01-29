# https://www.gnu.org/software/make/manual/

.DEFAULT_GOAL := .venv

SHELL = bash

requirements-dev.txt:
	# TODO: Bump pip-tools to >=7.0.0. See https://github.com/jazzband/pip-tools/issues/1659
	# https://github.com/jazzband/pip-tools#updating-requirements
	pip-compile --upgrade --extra=dev --output-file=requirements-dev.txt pyproject.toml

sync: requirements-dev.txt
	pip-sync requirements-dev.txt

install: sync
	python3 -m pip install --editable .

lint:
	mypy src/deeprl tests; \
	hatch run style:check

format:
	hatch run style:format

test:  # runs tests on every Python version with hatch
	hatch run test:cov

clean-pycache:  # removes Python file artifacts https://en.wikipedia.org/wiki/Artifact_(software_development)
	find . -type d -name '__pycache__' -exec rm -fr {} +
	find . -type d -name 'outputs' -exec rm -fr {} +
	find . -name '*~' -exec rm -f {} +
	rm -fr .mypy_cache

clean-test:  # removes test and coverage artifacts
	rm -fr .pytest_cache
	rm -f .coverage

clean-build:  # removes build artifacts
	rm -fr dist/

clean: clean-pycache clean-test clean-build

.venv:
	python3 -m venv .venv --clear
	source .venv/bin/activate; \
	python -m pip install --upgrade pip pip-tools pre-commit; \
	pre-commit install
