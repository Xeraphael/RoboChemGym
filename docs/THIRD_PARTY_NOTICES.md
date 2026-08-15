# Third-Party Notices

RoboChemGym is MIT licensed except where a source file carries another notice.

| Component | Location | License |
|---|---|---|
| Action Chunking Transformer (ACT) | `policy/model/act/` | MIT; derived from [tonyzhaozh/act](https://github.com/tonyzhaozh/act) |
| DETR components used by ACT | `policy/model/act/` | Apache-2.0; derived from [facebookresearch/detr](https://github.com/facebookresearch/detr) |
| imagecodecs numcodecs adapter | `policy/codecs/imagecodecs_numcodecs.py` | BSD-3-Clause; notice retained in the file |

The Apache-2.0 and ACT MIT license texts are included under `LICENSES/`.

NVIDIA Isaac Sim is an external runtime and is not covered by this repository's
MIT license. Users must obtain Isaac Sim and its assets separately and comply
with NVIDIA's applicable license terms. The project Franka wrapper imports the
robot implementation from the installed Isaac Sim runtime rather than
redistributing NVIDIA's implementation.

Legacy lab scenes and instrument packages containing non-redistributable NVIDIA
MDL sources are intentionally excluded. The public capability registry therefore
supports only assets embedded in the bundled `example_protocol` scene.
