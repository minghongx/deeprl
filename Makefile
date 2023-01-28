# https://www.gnu.org/software/make/manual/

.DEFAULT_GOAL := .venv

SHELL = bash

sources = src/deeprl tests

install: sync
	python3 -m pip install --editable .

rqmts:
	# https://github.com/jazzband/pip-tools/issues/1659
	pip-compile --resolver=backtracking --extra=dev --output-file=requirements.txt pyproject.toml

sync:
	pip-sync requirements.txt

test:  # runs tests on every Python version with hatch
	hatch run test:cov

clean: clean-pycache clean-test clean-build

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

if-in-venv:
ifndef VIRTUAL_ENV
	$(error This recipe should be executed in a virtual environment)
endif

format: if-in-venv
	isort $(sources)
	black $(sources)

lint: if-in-venv
	@ruff $(sources)
	@isort $(sources) --check-only --df
	@black $(sources) --check --diff

mypy: if-in-venv
	mypy src/deeprl

build: if-in-venv
	python -m build

.venv:
	python3 -m venv .venv --clear
	source .venv/bin/activate; \
	python -m pip install --upgrade pip pip-tools pre-commit; \
	pre-commit install
