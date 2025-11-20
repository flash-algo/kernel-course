# Contributing to kernel-course

Everyone is welcome to contribute, and we value everybody's contribution. Code contributions are not the only way to help the project – issues, discussions, and documentation improvements are also immensely valuable.

However you choose to contribute, please be mindful and respect our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Ways to contribute

There are several ways you can contribute to **kernel-course**:

* Fix outstanding issues with the existing kernels or tests.
* Submit issues related to bugs or desired new features.
* Implement new BLAS-style kernels (e.g. `scal`, `axpby`, `dot`, `gemv`, ...) in Python, PyTorch, Triton, or CuTe.
* Improve existing implementations for better readability, numerical stability, or performance.
* Contribute to the docs in `docs/` or examples/notebooks that use the kernels.

> All contributions are equally valuable to the project. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

### Did you find a bug?

Before you report an issue, please **check existing issues** to see if it has already been reported or fixed.

In a bug report, it helps a lot if you can include:

* Your **OS type and version** and **Python**, **PyTorch**, and **CUDA** versions.
* Your **GPU model** (if using CUDA/MPS).
* The exact command or script you ran (e.g. `pytest tests/test_copy.py -k "cuda"`).
* A short, self-contained code snippet that reproduces the bug.
* The *full* traceback if an exception is raised.

To get environment information quickly, you can run:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"
```

### Do you want a new feature?

If there is a new feature you'd like to see in kernel-course (e.g. a new kernel, backend, or tutorial), please open an issue and describe:

1. The *motivation* behind this feature (e.g. learning goal, missing building block).
2. The feature you are proposing (what kernel/module/backends you’d like to add).
3. Any references (papers, blog posts, repos) that inspired it.
4. Example usage if applicable.

## Do you want to add documentation?

Documentation PRs are very welcome. You can:

* Improve the main `README.md`.
* Add or refine docs under `docs/` for each kernel.
* Add comments or explanations in notebooks or example scripts.

## Create a Pull Request

Before writing any code, we strongly advise you to search through the existing PRs or issues to make sure nobody is already working on the same thing.

You will need basic `git` proficiency to contribute to kernel-course. You'll need **Python 3.9+** and a recent **PyTorch** installation; CUDA is required only if you want to run GPU backends.

### Development Setup

1. Fork the [repository](https://github.com/flash-algo/kernel-course) by clicking on the **Fork** button.

2. Clone your fork to your local disk, and add the base repository as a remote:

   ```bash
   git clone https://github.com/<your Github handle>/kernel-course.git
   cd kernel-course
   git remote add upstream https://github.com/flash-algo/kernel-course.git
   ```

3. Create a new branch to hold your development changes:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch!

4. Set up a development environment:

   ```bash
   # (Optional) create a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac

   # Install in development mode
   pip install -e .[dev]
   ```

5. Develop the features in your branch.

   As you work on your code, you should make sure the test suite passes. For quick, focused checks, you can run only the tests impacted by your changes, for example:

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

   To keep the codebase consistent, kernel-course relies on **black** and **ruff** (via `ruff format` and `ruff check`) for formatting and linting. After you make changes, you can apply automatic style fixes and basic checks only on the files modified on your branch with:

   ```bash
   make fixup
   ```

   This target is optimized to operate on Python files that differ from `main`.

   If you prefer to run style fixes on the whole codebase at once, use:

   ```bash
   make style
   ```

   Before opening a pull request, we recommend running the full quality gate:

   ```bash
   make quality
   ```

   which will run linting (`ruff check`), formatting checks (`ruff format --check`), and the test suite (`pytest tests`).

6. Once you're happy with your changes, add changed files using `git add` and record your changes with `git commit`:

   ```bash
   git add .
   git commit -m "A descriptive commit message"
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

7. Go to your fork on GitHub and click on **Pull Request** to open a pull request.

### Pull request checklist

- [ ] The pull request title should summarize your contribution.<br>
- [ ] If your pull request addresses an issue, please mention the issue number in the pull request description to make sure they are linked.<br>
- [ ] To indicate a work in progress please prefix the title with `[WIP]`.<br>
- [ ] Make sure existing tests pass.<br>
- [ ] If adding a new feature, also add tests for it.<br>
- [ ] All public functions have clear docstrings (where applicable).<br>
- [ ] If you changed performance-critical code, optionally share simple benchmark numbers (device, dtype, n).<br>

### Tests

We use `pytest` for testing. From the root of the repository, run:

```bash
pytest tests/ -v
```

You are encouraged to add tests alongside new kernels under `tests/` and keep them fast and focused.

### Code Style

We follow standard Python code style guidelines:

* Use descriptive variable names
* Add type hints where applicable
* Follow PEP 8 guidelines
* Add docstrings to all public functions

## Security

If you discover a security vulnerability, please send an e-mail to the maintainers. All security vulnerabilities will be promptly addressed.

## Questions?

If you have questions about contributing, feel free to open an issue or discussion in this repository.

Thank you for contributing to kernel-course! 🚀
