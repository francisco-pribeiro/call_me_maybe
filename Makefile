install:
	uv sync

run:
	uv run python -m src

run-large:
	uv run python -m src --model=Qwen/Qwen3-1.7B

run-verbose:
	uv run python -m src --verbose

run-large-verbose:
	uv run python -m src --model=Qwen/Qwen3-1.7B --verbose

run-compare:
	uv run python -m src --compare

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

fclean: clean
	rm -rf data/output
	rm -rf .venv

lint:
	uv run flake8 --exclude=.venv,llm_sdk,tests,moulinette .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude 'llm_sdk|moulinette'
