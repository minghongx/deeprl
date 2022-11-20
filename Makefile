# https://www.gnu.org/software/make/manual/html_node/Phony-Targets.html

.DEFAULT_GOAL := venv

SHELL = bash

venv_name = .venv
torch_repo = --extra-index-url=https://download.pytorch.org/whl/cu117

install:
	# https://setuptools.pypa.io/en/latest/userguide/development_mode.html
	python3 -m pip install $(torch_repo) --editable .

test:
	hatch run test:cov

if-in-venv:
ifndef VIRTUAL_ENV
	$(error This recipe should be executed in a virtual environment)
endif

rqmts: if-in-venv
	pip-compile $(torch_repo) --output-file=requirements.txt pyproject.toml

dev-rqmts: if-in-venv
	# https://github.com/jazzband/pip-tools/issues/1659
	pip-compile $(torch_repo) --resolver=backtracking --extra=dev --output-file=dev-requirements.txt pyproject.toml

all-rqmts: if-in-venv
	pip-compile $(torch_repo) --all-extras --output-file=all-requirements.txt pyproject.toml

sync: if-in-venv
	$(error Not implemented)

build: if-in-venv
	python -m build

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

.ONESHELL:
venv:
	# Create the venv if it does not exist
	test -d $(venv_name) || virtualenv --python `which python3.8` $(venv_name)
	source $(venv_name)/bin/activate
	python -m pip install --upgrade pip
	python -m pip install virtualenv pip-tools pre-commit
	pre-commit install
