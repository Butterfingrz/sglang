# Repository Guidelines

## Project Structure & Module Organization
- `python/sglang/`: Python package (frontend `lang/`, runtime `srt/`, evaluation `eval/`); tests in `python/sglang/test/`.
- `test/`: integration and runtime tests (pytest; `pytest.ini` sets `asyncio_mode=auto`).
- `sgl-router/`: Rust-based request router with Python bindings (`py_src/`); tests in `sgl-router/tests/` and `sgl-router/py_test/`.
- `sgl-kernel/`: CUDA/C++ kernels and wheels; tests in `sgl-kernel/tests/`.
- `docs/`, `examples/`, `benchmark/`, `scripts/`, `docker/`, `assets/`: supporting materials and tooling.

## Build, Test, and Development Commands
- Install for dev: `pip3 install -e python[dev]`
- Format Python: `make format` or run pre-commit (see below)
- Core tests: `pytest -q test python/sglang/test`
- Router: `make -C sgl-router test` (Rust) or `cargo test` in `sgl-router/`
- Kernels (only if you touch them): `make -C sgl-kernel build`; unit tests: `pytest -q sgl-kernel/tests`
- Quick local sanity check:
```bash
python3 -m sglang.launch_server --model Qwen/Qwen2-7B-Instruct
python3 -m sglang.test.few_shot_gsm8k --num-questions 50
```

## Coding Style & Naming Conventions
- Python: 4-space indent; format with Black; import order with isort (`profile=black`); lint (Ruff) and spell-check (codespell) via pre-commit.
- C++/CUDA: clang-format (style files in repo); keep kernels minimal and well-commented.
- Rust (router): `cargo fmt` and `cargo clippy` (`make -C sgl-router fmt check`).
- Names: modules/files `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Use pytest; name tests `test_*.py`; prefer fast, deterministic cases and mark GPU-heavy tests clearly.
- Run subsets with `pytest -k <expr>`; keep per-file runtime reasonable to avoid CI timeouts.

## Commit & Pull Request Guidelines
- Commits: prefer Conventional Commits (e.g., `feat:`, `fix:`, `docs:`) and optional scopes like `[router]`, `[AMD]` seen in history. Keep subject ≤72 chars and include brief rationale.
- Pre-flight: `pip3 install pre-commit && pre-commit install && pre-commit run --all-files` (fix issues, rerun until clean).
- PRs: describe what/why/how, include test plan, hardware context, and perf/accuracy deltas when relevant; link issues (`#1234`); add the `run-ci` label to trigger CI.

## Security & Configuration Tips
- Do not commit secrets or large model files; `.pre-commit` includes key and size checks.
- Prefer the dev Docker in docs for consistent CUDA/driver toolchains.
