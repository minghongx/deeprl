- [ ] Fix `pip-compile gym[classic_control] -> gym[classic-control]`. See https://github.com/jazzband/pip-tools/issues/1576 for more details. For now I just manually replace the hyphen with the undersocre in `dev-requirements.txt`.
- [x] pytest w/ [coverage](https://github.com/nedbat/coveragepy)
- [ ] [tox](https://github.com/tox-dev/tox)
- [x] [hatch](https://github.com/pypa/hatch)
- [ ] [autopep8](https://github.com/hhatto/autopep8)
- [x] [flake8](https://github.com/PyCQA/flake8)
- [x] [ruff](https://github.com/charliermarsh/ruff)
- [x] Use ruff alongside black
- [ ] Config [black](https://github.com/psf/black)
- [ ] Config [isort](https://github.com/PyCQA/isort)
- [ ] [pylint](https://github.com/PyCQA/pylint)
- [ ] Open help/browser by make like [this](https://github.com/jeshraghian/snntorch/blob/cd9f9c0cf36a31e73a55de03d2e1408a379be6c5/Makefile#L4)
- [ ] [Background](https://www.baeldung.com/linux/kill-background-process) demo, e.g.,
    ```makefile
    train.PID:
        python3 -m train & echo $$! > $@

    ann: train.PID

    stop: train.PID
        kill `cat $<` && rm $<
    ```
- [ ] Better *pre-commit* for Python project
    - https://github.com/pre-commit/pre-commit-hooks/blob/main/.pre-commit-config.yaml
    - https://github.com/pydantic/pydantic/blob/main/.pre-commit-config.yaml
    - https://gdevops.gitlab.io/tuto_git/tools/pre-commit/repos_hooks/black/black.html
- [ ] Better Makefile
    - https://github.com/pydantic/pydantic/blob/main/Makefile
    - https://n124080.medium.com/one-of-the-most-important-file-in-python-project-makefile-722e86e1c8ea
- [ ] Better sum tree
    - https://github.com/marcelpanzer/turtlebot3_machine_learning/blob/master/turtlebot3_dqn/src/turtlebot3_dqn/sumtree.py
- [ ] Test the sum tree
- [ ] Replace pytest-cov with coverage
