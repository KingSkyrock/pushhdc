# pushhdc
WIP

## development

```sh
uv sync                  # create .venv and install dev deps
uv run pytest            # run tests
uv run pytest -m "not slow"  # skip slow evolution tests
uv run ruff check        # lint
uv run ruff format       # format
uv run mypy              # type check
uv build                 # build sdist + wheel into dist/
```
