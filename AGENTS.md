# Repository Guidelines

## Project Structure & Module Organization
- `Home.py` starts the Streamlit app; keep heavy logic in `utils/`.
- `pages/` holds numbered Streamlit pages (prefix controls order); reuse shared components.
- `utils/` houses processing, clustering, GPU, and validation helpers with type hints; prefer extending existing modules.
- `pages/5_结构多样性评估.py` should stay UI-orchestration only; keep logic in:
  - `utils/structure_diversity_data.py` (流式读取/缓存/抽样)
  - `utils/structure_diversity_similarity.py` (相似性矩阵、k-NN、多样性统计)
  - `utils/structure_diversity_analysis.py` (降维与聚类分析)
  - `utils/structure_diversity_visualization.py` (图表渲染与理化性质分布对比)
- `configs/` stores pipeline YAML referenced by the UI and `scripts/run_pipeline.py`; comment new keys inline.
- `data/` is local-only; keep large datasets out of git. Diagnostics live in `test/`.

## Build, Test, and Development Commands
- `chmod +x create_env_step_by_step.sh && ./create_env_step_by_step.sh` provisions the conda/mamba workspace with step-by-step dependency installation (CPU/GPU aware).
- `conda activate CADD-Toolbox` and `streamlit run Home.py` launch the interactive UI.
- `python scripts/run_pipeline.py --input /path/to/molecules.sdf --config configs/default_config.yml --output runs/demo --use_gpu` runs the batch workflow; omit `--use_gpu` to stay on CPU.
- `bash check_cuda_version.sh` and `python test/check_gpu_support.py` confirm CUDA/GPU availability before GPU jobs; `python test/check_jax.py` is optional.

## Coding Style & Naming Conventions
- Code targets Python 3.10 with 4-space indentation, `snake_case` functions/modules, and `CamelCase` classes.
- Retain detailed Chinese docstrings and type hints describing RDKit objects and numerical expectations.
- Use module-level loggers and guard multiprocessing or Streamlit entry points with `if __name__ == "__main__":`.

## Lint & Format Requirements
- `ruff` and `black` are required gates for local changes and CI.
- Pin formatter/linter versions to avoid style drift between local and CI:
  - `ruff==0.15.4`
  - `black==24.2.0`
- Before commit, run:
  - `ruff check .`
  - `black --check --diff .`
- If checks fail, run:
  - `ruff check . --fix` (optionally `--unsafe-fixes` when needed and reviewed)
  - `black .`
- Re-run `ruff check .` and `black --check --diff .` until both pass.

## Testing Guidelines
- Place new files in `test/` and name them `test_<feature>.py`; expose helper functions so they can be imported.
- Reproduce GPU/CPU branches by running `python scripts/run_pipeline.py ...` on a trimmed dataset and capturing logs.
- Prefer quick smoke checks: `python test_environment.py`, `python test/check_gpu_support.py`, and optional `python test/check_jax.py`.
- Document fixtures or data slices needed for tests; keep checks fast so script-based smoke tests (and optional `pytest -q`) complete in minutes.

## Commit & Pull Request Guidelines
- Follow history conventions: concise subject (emoji or `feat:` prefix optional) plus details on performance or GPU impact.
- Reference issues (`Closes #123`), mention touched modules/configs, and update docs when behavior shifts.
- Pull requests must describe motivation, test evidence (commands + observed results), and attach Streamlit screenshots for UI tweaks.

## Configuration & Data Handling
- Version configuration templates under `configs/`; clone and adjust an existing file when adding pipelines.
- Keep `.streamlit/` secrets and large raw datasets outside git; provide download scripts or paths in documentation.
- When creating cache directories, update `.gitignore` and note the storage location in the relevant README or config comments.
