- [ ] Fix `pip-compile gym[classic_control] -> gym[classic-control]`. See https://github.com/jazzband/pip-tools/issues/1576 for more details. For now I just manually replace the hyphen with the undersocre in `dev-requirements.txt`.
- [ ] pytest w/ [coverage](https://github.com/nedbat/coveragepy)
- [ ] [tox](https://github.com/tox-dev/tox)
- [ ] [hatch](https://github.com/pypa/hatch)
- [ ] [autopep8](https://github.com/hhatto/autopep8)
- [ ] [flake8](https://github.com/PyCQA/flake8)
- [ ] Open help/browser by make like [this](https://github.com/jeshraghian/snntorch/blob/cd9f9c0cf36a31e73a55de03d2e1408a379be6c5/Makefile#L4)
- [ ] [Background](https://www.baeldung.com/linux/kill-background-process) demo, e.g.,
    ```makefile
    train.PID:
        python3 -m train & echo $$! > $@

    ann: train.PID

    stop: train.PID
        kill `cat $<` && rm $<
    ```
