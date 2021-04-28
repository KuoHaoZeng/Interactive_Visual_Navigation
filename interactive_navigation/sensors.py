from typing import Any, Dict, Optional, List, Tuple

import gym, quaternion, torch
import numpy as np

from interactive_navigation.environment import IThorEnvironment
from interactive_navigation.tasks import ObjectPlacementTask
from allenact.base_abstractions.sensor import Sensor, RGBSensor, DepthSensor
from allenact.base_abstractions.task import Task
from allenact.utils.misc_utils import prepare_locals_for_super
from interactive_navigation.utils.utils_3d_torch import (
    get_corners,
    local_project_2d_points_to_3d,
    project_2d_points_to_3d,
)


class RGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in iTHOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.current_frame.copy()


class LastRGBSensorThor(RGBSensor[IThorEnvironment, Task[IThorEnvironment]]):
    """Sensor for RGB images in iTHOR.

    Returns from a running IThorEnvironment instance, the current RGB
    frame corresponding to the agent's egocentric view.
    """

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.last_frame.copy()


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
        self,
        object_types: List[str],
        target_to_detector_map: Optional[Dict[str, str]] = None,
        detector_types: Optional[List[str]] = None,
        uuid: str = "goal_object_type_ind",
        **kwargs: Any
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }

            observation_space = gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            assert (
                detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

            observation_space = gym.spaces.Discrete(len(self.detector_types))

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
        self,
        env: IThorEnvironment,
        task: Optional[ObjectPlacementTask],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]


class GPSCompassSensorIThor(Sensor[IThorEnvironment, ObjectPlacementTask]):
    def __init__(self, uuid: str = "target_coordinates_ind", **kwargs: Any):
        observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def _compute_pointgoal(self, source_position, source_rotation, goal_position):
        direction_vector = goal_position - source_position
        direction_vector_agent = self.quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = GPSCompassSensorIThor.cartesian_to_polar(
            direction_vector_agent[2], -direction_vector_agent[0]
        )
        return np.array([rho, phi], dtype=np.float32)

    @staticmethod
    def quaternion_from_y_angle(angle: float) -> np.quaternion:
        r"""Creates a quaternion from rotation angle around y axis
        """
        return GPSCompassSensorIThor.quaternion_from_coeff(
            np.array(
                [0.0, np.sin(np.pi * angle / 360.0), 0.0, np.cos(np.pi * angle / 360.0)]
            )
        )

    @staticmethod
    def quaternion_from_coeff(coeffs: np.ndarray) -> np.quaternion:
        r"""Creates a quaternions from coeffs in [x, y, z, w] format
        """
        quat = np.quaternion(0, 0, 0, 0)
        quat.real = coeffs[3]
        quat.imag = coeffs[0:3]
        return quat

    @staticmethod
    def cartesian_to_polar(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    @staticmethod
    def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
        r"""Rotates a vector by a quaternion
        Args:
            quat: The quaternion to rotate by
            v: The vector to rotate
        Returns:
            np.array: The rotated vector
        """
        vq = np.quaternion(0, 0, 0, 0)
        vq.imag = v
        return (quat * vq * quat.inverse()).imag

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:

        agent_state = env.agent_state()
        agent_position = np.array([agent_state[k] for k in ["x", "y", "z"]])
        rotation_world_agent = self.quaternion_from_y_angle(
            agent_state["rotation"]["y"]
        )

        goal_position = np.array([task.task_info["target"][k] for k in ["x", "y", "z"]])

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )


class DepthSensorIThor(DepthSensor[IThorEnvironment, Task[IThorEnvironment]]):
    # For backwards compatibility
    def __init__(
            self,
            use_resnet_normalization: Optional[bool] = None,
            use_normalization: Optional[bool] = None,
            mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
            stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
            height: Optional[int] = None,
            width: Optional[int] = None,
            uuid: str = "depth",
            output_shape: Optional[Tuple[int, ...]] = None,
            output_channels: int = 1,
            unnormalized_infimum: float = 0.0,
            unnormalized_supremum: float = 5.0,
            scale_first: bool = False,
            **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.current_depth.copy()


class LastDepthSensorIThor(DepthSensor[IThorEnvironment, Task[IThorEnvironment]]):
    # For backwards compatibility
    def __init__(
            self,
            use_resnet_normalization: Optional[bool] = None,
            use_normalization: Optional[bool] = None,
            mean: Optional[np.ndarray] = np.array([[0.5]], dtype=np.float32),
            stdev: Optional[np.ndarray] = np.array([[0.25]], dtype=np.float32),
            height: Optional[int] = None,
            width: Optional[int] = None,
            uuid: str = "last_depth",
            output_shape: Optional[Tuple[int, ...]] = None,
            output_channels: int = 1,
            unnormalized_infimum: float = 0.0,
            unnormalized_supremum: float = 5.0,
            scale_first: bool = False,
            **kwargs: Any
    ):
        # Give priority to use_normalization, but use_resnet_normalization for backward compat. if not set
        if use_resnet_normalization is not None and use_normalization is None:
            use_normalization = use_resnet_normalization
        elif use_normalization is None:
            use_normalization = False

        super().__init__(**prepare_locals_for_super(locals()))

    def frame_from_env(self, env: IThorEnvironment, task: Optional[ObjectPlacementTask]) -> np.ndarray:
        return env.last_depth.copy()


class FrameSensorThor(Sensor):
    """Sensor for Class Segmentation in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    class segmentation corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 uuid="frame"):
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(height, width, 3),
            dtype=np.float64,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        return env.current_frame.copy()


class ClassSegmentationSensorThor(Sensor):
    """Sensor for Class Segmentation in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    class segmentation corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 uuid="seg"):
        self.objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(height, width, len(objectTypes)),
            dtype=np.float64,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        if not env.using_mask_rcnn:
            return env.get_masks_by_object_types(self.objectTypes).copy()
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            mask = np.ones((env.current_frame.shape[0], env.current_frame.shape[0])) * len(self.objectTypes)
            for idx, mask_rcnn_label in enumerate(labels):
                tmp = masks[idx]
                mask[np.where(tmp)] = mask_rcnn_label
            mask = mask.astype(np.float32)
            mask /= len(self.objectTypes)
            return np.expand_dims(mask, axis=2)


class LocalKeyPoints3DSensorThor(Sensor):
    """Sensor for Key Points of objects in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    key points of objects corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 uuid="class_segmentation"):
        self.objectTypes = objectTypes
        self.sorted_objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(len(objectTypes), 8, 3),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        key_points = []
        current_depth = env.current_depth
        current_depth = np.expand_dims(current_depth, axis=2)
        if not env.using_mask_rcnn:
            for objType in self.objectTypes:
                mask = env.get_mask_by_object_type(objType)
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = local_project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            for objType in self.objectTypes:
                mask_rcnn_label = self.sorted_objectTypes.index(objType)
                if mask_rcnn_label in labels:
                    idx = labels.index(mask_rcnn_label)
                    mask = masks[idx]
                else:
                    mask = np.zeros((env.current_frame.shape[0], env.current_frame.shape[1]))
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = local_project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])

        key_points = np.array(key_points, dtype=np.float32)
        return key_points


class GlobalKeyPoints3DSensorThor(Sensor):
    """Sensor for Key Points of objects in iTHOR.

    Returns from a running IThorEnvironment instance, the current
    key points of objects corresponding to the agent's egocentric view.
    """
    def __init__(self,
                 objectTypes,
                 uuid="class_segmentation"):
        self.objectTypes = objectTypes
        self.sorted_objectTypes = sorted(list(objectTypes))
        observation_space = gym.spaces.Box(
            low=-10,
            high=10,
            shape=(len(objectTypes), 8, 3),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        key_points = []
        current_depth = env.current_depth
        current_depth = np.expand_dims(current_depth, axis=2)
        if not env.using_mask_rcnn:
            for objType in self.objectTypes:
                mask = env.get_mask_by_object_type(objType)
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        else:
            output = env.get_mask_rcnn_result()
            labels = list(output["labels"].detach().cpu().numpy())
            masks = output["masks"].squeeze(1).detach().cpu().numpy()
            for objType in self.objectTypes:
                mask_rcnn_label = self.sorted_objectTypes.index(objType)
                if mask_rcnn_label in labels:
                    idx = labels.index(mask_rcnn_label)
                    mask = masks[idx]
                else:
                    mask = np.zeros((env.current_frame.shape[0], env.current_frame.shape[1]))
                points, depths = get_corners(mask, current_depth)
                points = torch.Tensor([points])
                depths = torch.Tensor([depths])
                points_3d = project_2d_points_to_3d([env.last_event.metadata], points, depths)
                key_points.append(points_3d.numpy()[0])
        key_points = np.array(key_points, dtype=np.float32)
        return key_points


class GlobalObjPoseSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_pose"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=-360,
            high=360,
            shape=(len(objectTypes), 6),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        vis_objects = env.visible_objects()
        vis_objects_type = [ele["objectType"] for ele in vis_objects]
        obj_pose = []
        for objType in self.objectTypes:
            if objType in vis_objects_type:
                idx = vis_objects_type.index(objType)
                pose = [vis_objects[idx]["position"]["x"],
                        vis_objects[idx]["position"]["y"],
                        vis_objects[idx]["position"]["z"],
                        vis_objects[idx]["rotation"]["x"],
                        vis_objects[idx]["rotation"]["y"],
                        vis_objects[idx]["rotation"]["z"]]
                obj_pose.append(pose)
            else:
                obj_pose.append([0, 0, 0, 0, 0, 0])
        return np.array(obj_pose, dtype=np.float32)


class GlobalObjUpdateMaskSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_update_mask"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(objectTypes), 1),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        vis_objects = env.visible_objects()
        vis_objects_type = [ele["objectType"] for ele in vis_objects]
        update_mask = []
        for objType in self.objectTypes:
            if objType in vis_objects_type:
                update_mask.append(1)
            else:
                update_mask.append(0)
        return np.array(update_mask, dtype=np.float32)


class GlobalObjActionMaskSensorThor(Sensor):
    def __init__(self,
                 objectTypes,
                 uuid="object_action_mask"):
        self.objectTypes = objectTypes
        observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(len(objectTypes), 1),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        obj = env.moveable_closest_obj_by_types(self.objectTypes)
        update_mask = [0] * len(self.objectTypes)
        if not isinstance(obj, type(None)):
            idx = self.objectTypes.index(obj["objectType"])
            update_mask[idx] = 1
        return np.array(update_mask, dtype=np.float32)


class GlobalAgentPoseSensorThor(Sensor):
    def __init__(self,
                 uuid="agent_pose"):
        observation_space = gym.spaces.Box(
            low=-360,
            high=360,
            shape=(6,),
            dtype=np.float32,
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectPlacementTask],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        agent_pose = [env.last_event.metadata["cameraPosition"]["x"],
                      env.last_event.metadata["cameraPosition"]["y"],
                      env.last_event.metadata["cameraPosition"]["z"],
                      env.last_event.metadata["agent"]["cameraHorizon"],
                      env.last_event.metadata["agent"]["rotation"]["y"],
                      0]
        return np.array(agent_pose, dtype=np.float32)

class MissingActionSensor(Sensor):
    def __init__(
            self,
            nactions: int,
            uuid: str = "missing_action",
            **kwargs: Any
    ) -> None:
        self.nactions = nactions
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(self.nactions + 1)


    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        missing_action = task.task_info["missing_action"]
        return missing_action

class MissingActionVectorSensor(Sensor):
    def __init__(
            self,
            nactions: int,
            uuid: str = "missing_action",
            **kwargs: Any
    ) -> None:
        self.nactions = nactions
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(self.nactions + 1)

    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        missing_action = task.task_info["missing_action"]
        out = np.zeros(self.nactions + 1)
        for ma in missing_action:
            out[ma] = 1
        return out


class MissingActionVectorMaskSensor(Sensor):
    def __init__(
            self,
            uuid: str = "missing_action_mask",
            **kwargs: Any
    ) -> None:
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Tuple:
        return gym.spaces.Discrete(1)

    def get_observation(
            self, env: IThorEnvironment, task: Optional[ObjectPlacementTask], *args: Any, **kwargs: Any
    ) -> Any:
        out = np.zeros(1)
        out[0] = task.missing_action_mask
        return out
