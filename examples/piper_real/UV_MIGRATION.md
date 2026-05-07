# uv Project Workflow

`examples/piper_real` now has its own `pyproject.toml`, so the recommended workflow is to rebuild a fresh uv-managed environment instead of copying the legacy `.venv`.

## What uv manages

- Python packages declared in `pyproject.toml`
- A project lockfile (`uv.lock`) after `uv lock`
- A dedicated environment such as `.venv-uv`

## What uv does not manage

- ROS itself
- ROS Python modules exposed through `source /opt/ros/<distro>/setup.bash`
- Interbotix and other robot-specific ROS packages
- NVIDIA driver / CUDA runtime compatibility

## Create `.venv-uv` in this directory

Run from `examples/piper_real`:

```bash
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --python 3.10.18
```

Run from the repo root with the same result:

```bash
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=examples/piper_real/.venv-uv uv sync --project examples/piper_real --python 3.10.18
```

## Run the example with uv

From `examples/piper_real`:

```bash
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv run python main.py
```

From the repo root:

```bash
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=examples/piper_real/.venv-uv uv run --project examples/piper_real python examples/piper_real/main.py
```

## Lock and reproduce the environment

After validating the environment on the source machine:

```bash
cd /home/agilex/rhos_cobot/examples/piper_real
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv lock --python 3.10.18
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --frozen --python 3.10.18
```

Check `uv.lock` into version control. Then on another Linux machine:

```bash
git clone <your-repo>
cd /path/to/repo/examples/piper_real
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --frozen --python 3.10.18
```

## Migration checklist for another Linux device

1. Keep the same CPU architecture. This environment is currently x86_64-oriented.
2. Source the same ROS distribution and any robot workspace before `uv run`.
3. Keep GPU driver and CUDA compatibility aligned with `torch==2.7.0`.
4. Ensure the target machine can access `git@gitee.com:yuz1wan/cobot_magic.git`, because `openpi-client` is resolved from that repository by default.
5. Prefer `uv sync --frozen` over copying `.venv`, because the old environment contains machine-specific paths.

## Optional: switch `openpi-client` to a local editable checkout

If you keep a local checkout of `openpi-client`, replace the Git source with a local editable source:

```bash
cd /home/agilex/rhos_cobot/examples/piper_real
uv remove openpi-client
uv add --editable /path/to/openpi/packages/openpi-client
```

That will update `pyproject.toml` for local development on the current machine.
