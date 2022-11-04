.PHONY: venv install clean rqmts dev-rqmts all-rqmts sync build
.ONESHELL:

SHELL = bash
venv_name = .venv
torch_repo = --extra-index-url=https://download.pytorch.org/whl/cu116

venv:
	# Create the venv if it does not exist
	test -d $(venv_name) || virtualenv --python `which python3.10` $(venv_name)
	source $(venv_name)/bin/activate
	python -m pip install --upgrade pip
	python -m pip install pip-tools

install:
	# https://setuptools.pypa.io/en/latest/userguide/development_mode.html
	python3 -m pip install $(torch_repo) --editable .

clean:
	rm -rf $(venv_name)
	find -iname "*.pyc" -delete

if-in-venv:
ifndef VIRTUAL_ENV
	$(error This recipe should be executed in a virtual environment)
endif

rqmts: if-in-venv
	pip-compile $(torch_repo) --output-file=requirements.txt pyproject.toml

dev-rqmts: if-in-venv
	pip-compile $(torch_repo) --extra=dev --output-file=dev-requirements.txt pyproject.toml

all-rqmts: if-in-venv
	pip-compile $(torch_repo) --all-extras --output-file=all-requirements.txt pyproject.toml

sync: if-in-venv
	$(error Not implemented)

build: if-in-venv
	python -m build
