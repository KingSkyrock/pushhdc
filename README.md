# PushHDC
Evolving hyperdimensional computing (HDC) encoders as Push programs with genetic programming.

## Development

```sh
uv sync                      # create .venv and install dev deps
uv run pytest                # run tests
uv run pytest -m "not slow"  # skip slow evolution tests
uv run ruff check            # lint
uv run ruff format           # format
uv run mypy                  # type check
uv build                     # build sdist + wheuv pip show pyshgpel into dist/
```

#### Setup / Updating

- `uv sync`: Run to create a virtual environment (`.venv`) from the dependencies in `pyproject.toml` and `uv.lock`. Must be run during initial setup or when dependencies change.

#### Development

- `uv run pytest -m "not slow"`: Run during development. Does not include longer evolution tests.
- `uv run ruff check`: Checks for lint issues, such as (bugs, unused imports).
- `uv run ruff format` Checks the code follows style conventions.
- `uv run mypy`: Use to check types in `src` and `tests` according to `pyproject.toml`.

#### Pushing

- `uv run pytest`: Run the full test suite for a complete check (recommended before pushing).

#### Distribution

- `uv build`: Builds both a source distribution and a wheel into `dist/`.
