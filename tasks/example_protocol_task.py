import numpy as np

from .all_task import AllTask


class ExampleProtocolTask(AllTask):
    """Bundled example scene with a return target at the source location."""

    def reset(self):
        super().reset()
        source_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.task.return_object_path
        )
        target_position = self.object_utils.get_object_xform_position(
            object_path=self.cfg.task.return_target_path
        )
        aligned_position = np.asarray(target_position).copy()
        aligned_position[:2] = np.asarray(source_position)[:2]
        self.object_utils.set_object_position(
            object_path=self.cfg.task.return_target_path,
            position=aligned_position,
        )
        if self.current_randomization is not None:
            constraints = self.current_randomization["resolved"].setdefault(
                "constraints", {}
            )
            constraints["return_target_alignment"] = {
                "source": self.cfg.task.return_object_path,
                "target": self.cfg.task.return_target_path,
                "resolved_position": aligned_position.tolist(),
            }
