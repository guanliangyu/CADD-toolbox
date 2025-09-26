# Repository Guidelines

## Project Structure & Module Organization
- `Home.py` starts the Streamlit app; keep heavy logic in `utils/`.
- `pages/` holds numbered Streamlit pages (prefix controls order); reuse shared components.
- `utils/` houses processing, clustering, GPU, and validation helpers with type hints; prefer extending existing modules.
- `configs/` stores pipeline YAML referenced by the UI and `scripts/run_pipeline.py`; comment new keys inline.
- `data/` is local-only; keep large datasets out of git. Diagnostics live in `test/`.

## Build, Test, and Development Commands
- `chmod +x create_env_step_by_step.sh && ./create_env_step_by_step.sh` provisions the conda/mamba workspace defined in `environment.yml`.
- `conda activate CADD-Toolbox` and `streamlit run Home.py` launch the interactive UI.
- `python scripts/run_pipeline.py --input data/sample.sdf --config configs/default_config.yml --output runs/demo --use_gpu` runs the batch workflow; omit `--use_gpu` to stay on CPU.
- `bash check_cuda_version.sh` and `python test/check_jax.py` confirm CUDA availability before GPU jobs.

## Coding Style & Naming Conventions
- Code targets Python 3.10 with 4-space indentation, `snake_case` functions/modules, and `CamelCase` classes.
- Retain detailed Chinese docstrings and type hints describing RDKit objects and numerical expectations.
- Use module-level loggers and guard multiprocessing or Streamlit entry points with `if __name__ == "__main__":`.

## Testing Guidelines
- Place new files in `test/` and name them `test_<feature>.py`; expose helper functions so they can be imported.
- Reproduce GPU/CPU branches by running `python scripts/run_pipeline.py ...` on a trimmed dataset and capturing logs.
- Document fixtures or data slices needed for tests; target quick checks so `pytest -q` (or direct script execution) completes in minutes.

## Commit & Pull Request Guidelines
- Follow history conventions: concise subject (emoji or `feat:` prefix optional) plus details on performance or GPU impact.
- Reference issues (`Closes #123`), mention touched modules/configs, and update docs when behavior shifts.
- Pull requests must describe motivation, test evidence (commands + observed results), and attach Streamlit screenshots for UI tweaks.

## Configuration & Data Handling
- Version configuration templates under `configs/`; clone and adjust an existing file when adding pipelines.
- Keep `.streamlit/` secrets and large raw datasets outside git; provide download scripts or paths in documentation.
- When creating cache directories, update `.gitignore` and note the storage location in the relevant README or config comments.
