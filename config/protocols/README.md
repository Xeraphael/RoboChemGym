# Planned protocol manifests

This directory contains the 20 protocol manifests associated with
[RoboChemGym: A Protocol-Driven Generative Simulation Framework for Long-Horizon Chemical Manipulation](https://arxiv.org/abs/2610.02708).
The paper record is also available as [arXiv:2610.02708](https://arxiv.org/abs/2610.02708).

These YAML files are **generation plans**, not executable Isaac Sim tasks. Every
manifest keeps the source description, materials, listed assets, atomic action
sequence, and the questions that must be resolved before scene generation.
`status: planned` and `executable: false` are intentional.

## Manifest schema

Each file contains:

- `source`: paper metadata, the original protocol description, and the supplied
  atomic action sequence;
- `assets`: asset IDs, roles, and whether each item is already in the capability
  registry;
- `materials`: quantities, phases, and source-container hints from the protocol;
- `atomic_actions`: normalized action tokens from the supplied sequence;
- `generation`: readiness status, pending capabilities, and open questions.

The current registry has `pick`, `place`, `pour`, `press`, `press_z`, `shake`,
`open`, and `close`. The two bottle types used by the protocols are marked
`pending_asset`, and weighing is marked as a pending capability because there is
no `weigh` action yet.

## Protocol index

| ID | Protocol | Assets | Actions |
|---|---|---|---|
| [protocol_01](protocol_01_heat_plate_placement.yaml) | Heat Plate Placement | pending | ready |
| [protocol_02](protocol_02_weigh_p_toluidine.yaml) | Weigh p-Toluidine | pending | pending |
| [protocol_03](protocol_03_add_pyridine_to_rbf.yaml) | Add Pyridine to RBF | pending | ready |
| [protocol_04](protocol_04_measure_ph.yaml) | Measure pH | ready | ready |
| [protocol_05](protocol_05_gently_shake_erlenmeyer.yaml) | Gently Shake Erlenmeyer | ready | ready |
| [protocol_06](protocol_06_oven_drying_setup.yaml) | Oven Drying Setup | ready | ready |
| [protocol_07](protocol_07_pour_into_suction_flask.yaml) | Pour into Suction Flask | ready | ready |
| [protocol_08](protocol_08_shaker_mixing.yaml) | Shaker Mixing | ready | ready |
| [protocol_09](protocol_09_ir_spectral_scan.yaml) | IR Spectral Scan | ready | ready |
| [protocol_10](protocol_10_stir_and_add_sulfuric_acid.yaml) | Stir and Add Sulfuric Acid | pending | ready |
| [protocol_11](protocol_11_add_methanol_after_weighing.yaml) | Add Methanol after Weighing | pending | pending |
| [protocol_12](protocol_12_charge_salicylic_acid_and_acetic_anhydride.yaml) | Charge Salicylic Acid and Acetic Anhydride | pending | pending |
| [protocol_13](protocol_13_carbonate_quench_and_ph_adjust.yaml) | Carbonate Quench and pH Adjust | ready | ready |
| [protocol_14](protocol_14_heat_and_shake_erlenmeyer.yaml) | Heat and Shake Erlenmeyer | ready | ready |
| [protocol_15](protocol_15_oven_dry_reaction_solution.yaml) | Oven Dry Reaction Solution | pending | ready |
| [protocol_16](protocol_16_weigh_and_transfer_benzoic_acid.yaml) | Weigh and Transfer Benzoic Acid | pending | pending |
| [protocol_17](protocol_17_bicarbonate_neutralization_shake.yaml) | Bicarbonate Neutralization Shake | pending | ready |
| [protocol_18](protocol_18_esterification_charge_sequence.yaml) | Esterification Charge Sequence | pending | pending |
| [protocol_19](protocol_19_heat_shake_transfer_oven_dry.yaml) | Heat, Shake, Transfer, Oven Dry | ready | ready |
| [protocol_20](protocol_20_dissolve_and_acid_adjust_sodium_acetate.yaml) | Dissolve and Acid-Adjust Sodium Acetate | pending | pending |

## Validation

From the repository root, parse every manifest with Ruby's YAML parser:

```bash
ruby -e 'require "yaml"; Dir["config/protocols/*.yaml"].each { |f| YAML.load_file(f); puts f }'
```

Before converting a manifest into an executable protocol bundle, resolve every
item listed under `generation.open_questions`, add the missing assets/actions to
the registry, and generate a scene, plan, validation report, and execution
configuration.
