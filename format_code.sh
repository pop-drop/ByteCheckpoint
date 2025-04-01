ruff format --config pyproject.toml
ruff check . --select I --fix
ruff check .
ruff check . --fix --unsafe-fixes