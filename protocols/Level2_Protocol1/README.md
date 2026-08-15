# Level2_Protocol1

This bundle records the validated six-step benzoic-acid dissolution protocol:

1. Pick the solid flask.
2. Place it on the heating plate.
3. Press the heating plate control.
4. Pick the liquid flask.
5. Pour into the solid flask.
6. Return the liquid flask to its original location.

`agent_plan.json` is the executable structured plan. `validation_report.json`
and `scene_preflight.json` are the pre-execution validation artifacts.
`scene.usd` is the exact scene used by `config/Level2_Protocol1.yaml`.

Run from the repository root:

```bash
./scripts/run_level2_protocol1.sh
```

The configuration uses `mode: execute`: it writes the execution report,
trajectory, and attempt video under `outputs/Level2_Protocol1/`, but does not
write an HDF5 dataset.
