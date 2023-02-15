- [ ] mypy and ruff should not lint based on the dev rqmts
- [ ] `__all__` in each module
    - https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html
- [ ] Replace pytest-cov with [coverage](https://github.com/nedbat/coveragepy)
- [ ] Open help/browser by make like [this](https://github.com/jeshraghian/snntorch/blob/cd9f9c0cf36a31e73a55de03d2e1408a379be6c5/Makefile#L4)
    - https://linux.die.net/man/1/xdg-open
- [ ] Set up CI on GitHub
- [ ] Figure out how to build with Hatch
- [ ] Move the demos into integration tests
- [ ] Better clean phony targets
- [ ] Add class docstring
    - https://stackoverflow.com/a/69671835/20015297
- [ ] Implement a custom `__subclasshook__()` method that allows runtime structural checks without explicit registration
    - [Make protocols special objects at runtime rather than normal ABCs](https://peps.python.org/pep-0544/#make-protocols-special-objects-at-runtime-rather-than-normal-abcs)


### Faster experience replay
- [Pre-allocated and memory-mapped experience replay](https://discuss.pytorch.org/t/rfc-torchrl-replay-buffers-pre-allocated-and-memory-mapped-experience-replay/155335)
- [segment tree C++ implementation](https://github.com/pytorch/rl/blob/main/torchrl/csrc/segment_tree.h)
    - [Write a C++ extension module for Python](https://opensource.com/article/22/11/extend-c-python)
    - https://mesonbuild.com/Comparisons.html
    - https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
- https://pytorch.org/rl/reference/generated/torchrl.data.PrioritizedReplayBuffer.html
- https://discuss.pytorch.org/t/how-to-make-the-replay-buffer-more-efficient/80986


### [JAX](https://github.com/deepmind/jax) impl
- [From PyTorch to JAX: towards neural net frameworks that purify stateful code](https://sjmielke.com/jax-purify.htm)
- https://github.com/ikostrikov/jaxrl2
- https://github.com/ikostrikov/jaxrl
- https://github.com/ikostrikov/walk_in_the_park/tree/main/rl
- [Using JAX to accelerate our research](https://www.deepmind.com/blog/using-jax-to-accelerate-our-research)
- [IsaacGym JAX integration](https://forums.developer.nvidia.com/t/isaacgym-jax-integration/228214/4)


### [Replace Conditional with Polymorphism](https://www.refactoring.com/catalog/replaceConditionalWithPolymorphism.html)

For example,
```python
import torch.nn as nn

net = nn.Sequential(
    *([nn.Conv2d(4, 4, 8)] if True else []),
    *([nn.Conv2d(4, 4, 8)] if False else []),
    nn.ReLU()
)
```
The above code uses
- [Conditional expressions](https://peps.python.org/pep-0308/)
- [Iterable unpacking](https://docs.python.org/3/reference/expressions.html#expression-lists)
- [Why does Python allow unpacking an empty iterable?](https://stackoverflow.com/questions/67359996/why-does-python-allow-unpacking-an-empty-iterable)

to achieve concise dynamic configuration of the NN but lost sematics.


### Python 3.9
- [PEP 585](https://peps.python.org/pep-0585/) Type Hinting Generics


### Python 3.10
- [PEP 604](https://peps.python.org/pep-0604/) Writing `Union[X, Y]` as `X | Y`
- [PEP 643](https://peps.python.org/pep-0634/) Convert multiple isinstance checks to structural pattern matching


### Python 3.11
- [Self Type](https://peps.python.org/pep-0673/)
- [Variadic Generics](https://peps.python.org/pep-0646/)


---

- [x] Improve algo classes init readability by attrs
- [x] Replace ABC with Protocol
    - **[Rationale](https://peps.python.org/pep-0544/#rationale-and-goals)**
    - [Interface segregation principle](https://en.wikipedia.org/wiki/Interface_segregation_principle)
    - [Design by contract](https://en.wikipedia.org/wiki/Design_by_contract)
    - [Nominal](https://en.wikipedia.org/wiki/Nominal_type_system) vs [structural](https://en.wikipedia.org/wiki/Structural_type_system) type system
    - [Every class is a type](https://peps.python.org/pep-0483/#types-vs-classes)
    - [Existing Approaches to Structural Subtyping](https://peps.python.org/pep-0544/#existing-approaches-to-structural-subtyping)
        - > ABCs in `typing` module already provide **structural behavior at runtime**, isinstance(Bucket(), Iterable) returns True. The main goal of this proposal is to **support such behavior statically**.
    - [Abstract Base Classes and Protocols](https://jellis18.github.io/post/2022-01-11-abc-vs-protocol/)
    - > In object-oriented programming, an interface or protocol type is a data type describing a set of method signatures, the implementations of which may be provided by multiple classes that are otherwise not necessarily related to each other. A class which provides the methods listed in a protocol is said to adopt the protocol, or to implement the interface. From [Wikipedia](https://en.wikipedia.org/wiki/Interface_(object-oriented_programming)).
    - `isinstance()` with protocols is not completely safe at runtime.
        - [Support `isinstance()` checks by default](https://peps.python.org/pep-0544/#support-isinstance-checks-by-default)
        - `@typing.runtime_checkable` [doc](https://docs.python.org/3/library/typing.html?highlight=typing#typing.runtime_checkable)
        - [Using `isinstance()` with protocols](https://mypy.readthedocs.io/en/latest/protocols.html#using-isinstance-with-protocols)
    - Worth noting rejected/postponed idea in PEP 544
        - [Protocols subclassing normal classes](https://peps.python.org/pep-0544/#protocols-subclassing-normal-classes)
        - [Make every class a protocol by default](https://peps.python.org/pep-0544/#make-every-class-a-protocol-by-default)
        - [Prohibit explicit subclassing of protocols by non-protocols](https://peps.python.org/pep-0544/#prohibit-explicit-subclassing-of-protocols-by-non-protocols)
    - [Modules as implementations of protocols](https://peps.python.org/pep-0544/#modules-as-implementations-of-protocols)
    - [match/case by type of value](https://stackoverflow.com/a/72295907/20015297)
- [x] Solved. See https://github.com/Farama-Foundation/Gymnasium/pull/180#discussion_r1038995462. Fix `pip-compile gym[classic_control] -> gym[classic-control]`. See https://github.com/jazzband/pip-tools/issues/1576 for more details.
- [x] [hatch](https://github.com/pypa/hatch)
- [x] [flake8](https://github.com/PyCQA/flake8)
- [x] [ruff](https://github.com/charliermarsh/ruff)
- [x] Use ruff alongside black
- [x] Config [black](https://github.com/psf/black)
- [x] Config [isort](https://github.com/PyCQA/isort)
- [x] Better *pre-commit* for Python project
    - https://github.com/pre-commit/pre-commit-hooks/blob/main/.pre-commit-config.yaml
    - https://github.com/pydantic/pydantic/blob/main/.pre-commit-config.yaml
    - https://github.com/tiangolo/fastapi/blob/master/.pre-commit-config.yaml
    - https://gdevops.gitlab.io/tuto_git/tools/pre-commit/repos_hooks/black/black.html
- [x] Better Makefile
    - https://github.com/pydantic/pydantic/blob/main/Makefile
    - https://n124080.medium.com/one-of-the-most-important-file-in-python-project-makefile-722e86e1c8ea
- [x] Better mypy
    - https://github.com/pytorch/pytorch/blob/master/mypy.ini
