# uv Project Workflow

`examples/piper_real` now has its own `pyproject.toml`, so the recommended workflow is to rebuild a fresh uv-managed `.venv` instead of continuing to use a legacy hand-built environment.

## What uv manages

- Python packages declared in `pyproject.toml`
- A project lockfile (`uv.lock`) after `uv lock`
- The default dedicated environment, `.venv`

## What uv does not manage

- ROS itself
- ROS Python modules exposed through `source /opt/ros/<distro>/setup.bash`
- Interbotix and other robot-specific ROS packages
- NVIDIA driver / CUDA runtime compatibility

## OpenPI checkout

Keep OpenPI as a sibling checkout of `rhos_cobot` so this repository is not
coupled to `~/cobot_magic`:

```bash
cd /home/agilex
git clone yushun_github:YushunXiang/OCL-openpi.git openpi
```

`pyproject.toml` resolves `openpi-client` from
`../../../openpi/packages/openpi-client`, relative to `examples/piper_real`.
If you use a different OpenPI location, update the `openpi-client` entry in
`pyproject.toml` or run `uv add --editable /path/to/openpi/packages/openpi-client`.

## Create `.venv` in this directory

Run from `examples/piper_real`:

```bash
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.10.18
```

Run from the repo root with the same result:

```bash
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --project examples/piper_real --python 3.10.18
```

With `--project examples/piper_real`, relative `UV_PROJECT_ENVIRONMENT` values
are resolved from the project directory. Use `.venv` here, or use an absolute
path if you need to name the environment from the repo root.

## Run the example with uv

From `examples/piper_real`:

```bash
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=.venv uv run python main.py
```

From the repo root:

```bash
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=.venv uv run --project examples/piper_real python examples/piper_real/main.py
```

The deploy helper also uses `examples/piper_real/.venv/bin/python` by default,
so a uv-synced `.venv` works with `scripts/run_piper_deploy.sh` without changing
that script's runtime flow.

## Lock and reproduce the environment

After validating the environment on the source machine:

```bash
cd /home/agilex/rhos_cobot/examples/piper_real
source /opt/ros/<distro>/setup.bash
UV_PROJECT_ENVIRONMENT=.venv uv lock --python 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen --python 3.10.18
```

Check `uv.lock` into version control. Then on another Linux machine:

```bash
git clone <your-repo>
cd /path/to/repo/examples/piper_real
source /opt/ros/<distro>/setup.bash
uv python install 3.10.18
UV_PROJECT_ENVIRONMENT=.venv uv sync --frozen --python 3.10.18
```

## Migration checklist for another Linux device

1. Keep the same CPU architecture. This environment is currently x86_64-oriented.
2. Source the same ROS distribution and any robot workspace before `uv run`.
3. Keep GPU driver and CUDA compatibility aligned with `torch==2.7.0`.
4. Clone `yushun_github:YushunXiang/OCL-openpi.git` as a sibling `openpi` checkout, or update the local `openpi-client` path in `pyproject.toml`.
5. Prefer `uv sync --frozen` over copying `.venv`, because the old environment contains machine-specific paths.

## Optional: switch `openpi-client` to another checkout

If you keep OpenPI somewhere other than `/home/agilex/openpi`, replace the
editable source:

```bash
cd /home/agilex/rhos_cobot/examples/piper_real
uv remove openpi-client
uv add --editable /path/to/openpi/packages/openpi-client
```

That will update `pyproject.toml` for local development on the current machine.
