Contributing
============

As an open source project, Mesa-Geo welcomes contributions of many forms, and from beginners to experts. If you are curious or just want to see what is happening, we post our development session agendas and development session notes on [Mesa discussions]. We also have a threaded discussion forum on [Matrix] for casual conversation.

In no particular order, examples include:

- Code patches
- Bug reports and patch reviews
- New features
- Documentation improvements
- Tutorials

No contribution is too small. Although, contributions can be too big, so let's discuss via [Matrix] or via an [issue].

[Mesa discussions]: https://github.com/mesa/mesa/discussions
[Matrix]: https://matrix.to/#/#project-mesa:matrix.org
[issue]: https://github.com/mesa/mesa-geo/issues

**To submit a contribution**

- Create a ticket for the item that you are working on.
- Fork the Mesa-Geo repository.
- [Clone your repository] from GitHub to your machine.
- Create a new branch in your fork: `git checkout -b BRANCH_NAME`
- Run `git config pull.rebase true`. This prevents messy merge commits when updating your branch on top of Mesa-Geo main branch.
- Install an editable version with developer requirements locally: `pip install -e ".[dev]"`
- Edit the code. Save.
- Git add the new files and files with changes: `git add FILE_NAME`
- Git commit your changes with a meaningful message: `git commit -m "Fix issue X"`
- If implementing a new feature, include some documentation in docs folder.
- Make sure that your submission passes the [GH Actions build]. See "Testing and Standards below" to be able to run these locally.
- Make sure that your code is formatted according to the [Ruff] standard (you can do it via [pre-commit]).
- Push your changes to your fork on Github: `git push origin NAME_OF_BRANCH`.
- [Create a pull request].
- Describe the change w/ ticket number(s) that the code fixes.

[Clone your repository]: https://help.github.com/articles/cloning-a-repository/
[GH Actions build]: https://github.com/mesa/mesa-geo/actions/workflows/build_lint.yml
[Create a pull request]: https://help.github.com/articles/creating-a-pull-request/
[pre-commit]: https://github.com/pre-commit/pre-commit
[Ruff]: https://github.com/astral-sh/ruff

Testing and Code Standards
--------------------------

[![codecov badge](https://codecov.io/gh/projectmesa/mesa-geo/branch/main/graph/badge.svg)](https://codecov.io/gh/projectmesa/mesa-geo) [![code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

As part of our contribution process, we practice continuous integration and use GH Actions to help enforce best practices.

If you're changing previous Mesa-Geo features, please make sure of the following:

- Your changes pass the current tests.
- Your changes pass our style standards.
- Your changes don't break the models or your changes include updated models.
- Additional features or rewrites of current features are accompanied by tests.
- New features are demonstrated in a model, so folks can understand more easily.

To ensure that your submission will not break the build, you will need to install Ruff and pytest.

```bash
pip install ruff pytest pytest-cov
```

We test by implementing simple models and through traditional unit tests in the tests/ folder. The following only covers unit tests coverage. Ensure that your test coverage has not gone down. If it has and you need help, we will offer advice on how to structure tests for the contribution.

```bash
pytest --cov=mesa_geo tests/
```

With respect to code standards, we follow [PEP8] and the [Google Style Guide]. We recommend to use [Ruff] as an automated code formatter. You can automatically format your code using [pre-commit], which will prevent `git commit` of unstyled code and will automatically apply Ruff style so you can immediately re-run `git commit`. To set up pre-commit run the following commands:

```bash
pip install pre-commit
pre-commit install
```

You should no longer have to worry about code formatting: pre-commit will run checks automatically when you commit.

If you still want to run the same checks manually (recommended before opening a PR), run pre-commit across the entire repository:

```bash
pre-commit run --all-files
```

If you prefer to run Ruff directly (or if you want a quick autofix pass), you can also run:

```bash
ruff check . --fix
```

[PEP8]: https://www.python.org/dev/peps/pep-0008
[Google Style Guide]: https://google.github.io/styleguide/pyguide.html
[pre-commit]: https://github.com/pre-commit/pre-commit
[Ruff]: https://github.com/astral-sh/ruff

Licensing
---------

The license of this project is located in [LICENSE]. By submitting a contribution to this project, you are agreeing that your contribution will be released under the terms of this license.

[LICENSE]: https://github.com/mesa/mesa-geo/blob/main/LICENSE

Special Thanks
--------------

A special thanks to the following projects who offered inspiration for this contributing file.

- [Django](https://github.com/django/django/blob/master/CONTRIBUTING.rst)
- [18F's FOIA](https://github.com/18F/foia-hub/blob/master/CONTRIBUTING.md)
- [18F's Midas](https://github.com/18F/midas/blob/devel/CONTRIBUTING.md)
