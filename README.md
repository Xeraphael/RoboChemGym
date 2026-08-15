# RoboChemGym

**A Protocol-Driven Generative Simulation Framework for Long-Horizon Chemical
Manipulation**

[English](README.md) | [简体中文](README_CN.md)

![RoboChemGym overview](docs/images/robochemgym-teaser.png)

RoboChemGym turns natural-language chemistry protocols into executable robot
trajectories, scales those trajectories through lab-specific randomization, and
provides Isaac Sim policy deployment and evaluation entry points. It is built on the
LabUtopia simulation environment.

This repository is the first source release. It focuses on complete code paths
and reproducible configuration rather than publishing trained policy results.

## Overview

RoboChemGym contains three connected modules:

1. **Protocol-driven trajectory generation** converts text into a structured
   Agent Plan, validates actions and assets, builds the scene, executes in Isaac
   Sim, and records reports and sampled videos.
2. **Randomized data collection** varies supported scene factors and stores
   successful episodes through an atomic manifest workflow with deterministic
   dataset splits.
3. **Policy deployment and evaluation** checks checkpoint/environment
   compatibility, runs fixed-seed Isaac evaluation, and writes structured task
   and action-level metrics.

## Release Status

| Method module | Status |
|---|---|
| Protocol-driven trajectory generation | Released |
| Randomized data collection | Released |
| Policy deployment and evaluation | Released |

## Installation

Requirements:

- Ubuntu Linux
- Python 3.10
- NVIDIA GPU and compatible driver
- Isaac Sim `4.2.0.2`
- Git LFS

```bash
git clone https://github.com/Xeraphael/RoboChemGym.git
cd RoboChemGym
git lfs install
git lfs pull

conda create -n labutopia python=3.10 -y
conda activate labutopia
pip install "isaacsim[all,extscache]==4.2.0.2" \
  --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
```

## API Configuration

The Action Agent uses an OpenAI-compatible chat-completions endpoint. Configure
it only through environment variables; do not commit `.env`.

```bash
cp .env.example .env
set -a
source .env
set +a
```

The endpoint, model, API key, timeout, and retry limit are documented in
[`.env.example`](.env.example).

## Quick Start

### 1. Run a Protocol

Write the procedure you want to execute in a text file, then pass its path to
the Action Agent:

```bash
python agent/main.py --protocol path/to/protocol.txt
```

For example, run the bundled Protocol with:

```bash
python agent/main.py --protocol protocols/example_protocol/protocol.txt
```

The execution flow is:

```text
text Protocol
  -> LLM structured Agent Plan
  -> deterministic Validator
  -> USD/YAML compilation and scene preflight
  -> Isaac execution and state-based verification
  -> bounded parameter/scene retry
  -> report, trajectory, and sampled attempt videos
```

Each run receives an isolated directory under `outputs/action_agent/`. Resume a
validated plan without another LLM request:

```bash
python agent/main.py --resume outputs/action_agent/<run-id>
```

Missing assets or core atomic actions block execution. Unsupported descriptive
semantics that cannot be observed, such as "avoid overheating", are retained as
non-blocking warnings while the closest supported physical action executes.

The legacy code-generation backend executes LLM-generated Python. It remains
disabled unless `--allow-unsafe-codegen` is explicitly supplied and should only
be used in a disposable isolated environment.

The repository includes the corresponding reusable Plan, validation artifacts,
configuration, and scene under
[`protocols/example_protocol/`](protocols/example_protocol/).

### 2. Collect randomized episodes

```bash
python collect.py --config config/collection/example_protocol.yaml \
  --/rtx/verifyDriverVersion/enabled=false
```

The bundled configuration randomizes object position, light intensity and color
temperature, and existing work-surface materials. Resolved values are recorded
in the episode manifest. Episodes are first written as partial files and enter a
dataset split only after a successful close.

### 3. Deploy and evaluate

```bash
python evaluate.py --config config/evaluation/example_protocol_act.yaml \
  --/rtx/verifyDriverVersion/enabled=false
```

Evaluation runs are written to
`outputs/evaluation/example_protocol_act/runs/<run-id>/` with an explicit
`running`, `completed`, or `incomplete` status. Dataset identity, schema, camera
order and shapes, state/action conventions, and gripper convention must match
the checkpoint before policy control starts.

## Evaluation Outputs

The evaluation report supports:

| Metric | Description |
|---|---|
| Task success | Fraction of episodes completing all required actions |
| Per-step success | Success count for each ordered Protocol step |
| Per-action success | Aggregation by atomic action type |
| Failure taxonomy | Validation, asset, execution, timeout, and compatibility failures |
| Episode length | Executed control steps per episode |
| Terminal distance | Final object-to-target distance where observable |

No trained policy or policy performance claim is part of the source release.

## Code Structure

```text
agent/                    Protocol parsing, planning, validation, and execution
controllers/              Atomic actions and robot controllers
data_collectors/          Cameras, episode recording, and HDF5 collection
pipeline/                 Collection, training, and evaluation orchestration
policy/                   ACT model, datasets, runners, and workspaces
tasks/                    Isaac task definitions
protocols/                Reproducible Protocol bundles
config/                   Collection, training, and evaluation configurations
scripts/                  Reproducible launch commands
tests/                    Focused code-level validation
docs/                     Project images and third-party notices
```

## Assets

The public source snapshot intentionally contains only the assets embedded in
the bundled `example_protocol` scene: two Erlenmeyer flask variants, a heating
plate, and a target platform. Other capability-registry entries are retained for
compatibility but fail validation until redistributable assets are supplied and
their paths are updated.

Legacy proprietary lab scenes and the private instrument library are not part of
this release. Isaac Sim and its runtime assets must be installed separately.

## Isaac Sim Driver Verification

RoboChemGym remains pinned to Isaac Sim `4.2.0.2`. Project entry points inject
`--/rtx/verifyDriverVersion/enabled=false` to bypass the Kit 106.1 driver-version
parsing issue without disabling rendering or physics. Passing the argument
explicitly is also supported, as shown above.

## Citation

The paper citation will be added after publication. For software metadata, see
[`CITATION.cff`](CITATION.cff).

## License

RoboChemGym source code is released under the [MIT License](LICENSE). Third-party
code, Isaac Sim, and external assets retain their respective licenses; see
[Third-party notices](docs/THIRD_PARTY_NOTICES.md).
