# example_protocol

This directory contains one complete example bundle:

- `protocol.txt`: natural-language input for the Action Agent.
- `agent_plan.json`: validated structured action plan.
- `validation_report.json` and `scene_preflight.json`: pre-execution checks.
- `scene.usd`: example scene used by `config/example_protocol.yaml`.

Run the precompiled bundle from the repository root:

```bash
./scripts/run_example_protocol.sh
```

The command saves reports, trajectories, and videos under
`outputs/example_protocol/` without writing an HDF5 dataset.
