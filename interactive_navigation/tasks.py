import gym
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Sequence, cast

from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task
from allenact.utils.system import get_logger

from interactive_navigation.environment import IThorEnvironment
from interactive_navigation.constants import (
    MOVE_AHEAD,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
    DIRECTIONAL_AHEAD_PUSH,
    DIRECTIONAL_BACK_PUSH,
    DIRECTIONAL_RIGHT_PUSH,
    DIRECTIONAL_LEFT_PUSH,
)

class ObstaclesNavTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.last_tget_in_path = False

        self.optimal_distance = self.last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            angle = [0.001, 180, 90, 270][action - 5]
            obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
            if obj != None:
                self.env.step({"action": action_str,
                               "objectId": obj["objectId"],
                               "moveMagnitude": obj["mass"] * 100,
                               "pushAngle": angle,
                               "autoSimulation": False})
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False
        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        tget = self.task_info["target"]
        dist = self.dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.last_geodesic_distance
        if (
                self.last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.last_geodesic_distance - geodesic_distance
        self.last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def shaping_by_path(self) -> float:
        reward = 0.0
        if self.env.last_action in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                                    DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            tget_in_path = self.env.target_in_reachable_points(self.task_info["target"])
            if tget_in_path and not self.last_tget_in_path:
                reward += 0.5
            elif not tget_in_path and self.last_tget_in_path:
                reward -= 0.5
            self.last_tget_in_path = tget_in_path
        return reward

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()
        reward += self.shaping_by_path()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        return self.env.distance_to_point(self.task_info["target"])

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.dist_to_target()
        spl = self.spl()

        return {
            **super(ObstaclesNavTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "target_in_reachable_points": self.last_tget_in_path,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError


class ObjectPlacementTask(Task[IThorEnvironment]):
    _actions = (MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_DOWN, LOOK_UP,
                DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH, DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH,
                END)

    def __init__(
            self,
            env: IThorEnvironment,
            sensors: List[Sensor],
            task_info: Dict[str, Any],
            max_steps: int,
            reward_configs: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(
            env=env, sensors=sensors, task_info=task_info, max_steps=max_steps, **kwargs
        )
        self.reward_configs = reward_configs
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.last_geodesic_distance = self.env.distance_to_point(
            self.task_info["target"]
        )
        self.obj_last_geodesic_distance = self.obj_dist_to_target()
        self.last_both_in_path = False

        self.optimal_distance = self.last_geodesic_distance + self.obj_last_geodesic_distance
        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List[Any] = (
            []
        )  # the initial coordinate will be directly taken from the optimal path

        self.task_info["followed_path"] = [self.env.agent_state()]
        self.task_info["action_names"] = self.action_names()
        self.num_moves_made = 0

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self._actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    @classmethod
    def class_action_names(cls, **kwargs) -> Tuple[str, ...]:
        return cls._actions

    def close(self) -> None:
        self.env.stop()

    def _step(self, action: Union[int, Sequence[int]]) -> RLStepResult:

        assert isinstance(action, int)
        action = cast(int, action)

        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
        elif action_str in [DIRECTIONAL_AHEAD_PUSH, DIRECTIONAL_BACK_PUSH,
                            DIRECTIONAL_RIGHT_PUSH, DIRECTIONAL_LEFT_PUSH]:
            angle = [0.001, 180, 90, 270][action - 5]
            obj = self.env.moveable_closest_obj_by_types(self.task_info["obstacles_types"])
            if obj != None:
                self.env.step({"action": action_str,
                               "objectId": obj["objectId"],
                               "moveMagnitude": obj["mass"] * 100,
                               "pushAngle": angle,
                               "autoSimulation": False})
                self.last_action_success = self.env.last_action_success
            else:
                self.last_action_success = False
        elif action_str in [LOOK_UP, LOOK_DOWN]:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success
            pose = self.env.agent_state()
            self.path.append({k: pose[k] for k in ["x", "y", "z"]})
            self.task_info["followed_path"].append(pose)
        if len(self.path) > 1 and self.path[-1] != self.path[-2]:
            self.num_moves_made += 1
        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )

        return step_result

    def render(self, mode: str = "rgb", *args, **kwargs) -> np.ndarray:
        assert mode in ["rgb", "depth"], "only rgb and depth rendering is implemented"
        if mode == "rgb":
            return self.env.current_frame
        elif mode == "depth":
            return self.env.current_depth

    def _is_goal_in_range(self) -> Optional[bool]:
        objs = self.env.get_objects_by_type(self.task_info["object_type"])
        tgt_obj = self.env.get_objects_by_type(self.task_info["target_type"])[0]
        for obj in objs:
            if obj["objectId"] in tgt_obj["receptacleObjectIds"]:
                return True

        tget = self.task_info["target"]
        dist = self.obj_dist_to_target()

        if -0.5 < dist <= 0.2:
            return True
        elif dist > 0.2:
            return False
        else:
            get_logger().debug(
                "No path for {} from {} to {}".format(
                    self.env.scene_name, self.env.agent_state(), tget
                )
            )
            return None

    def shaping(self) -> float:
        rew = 0.0

        if self.reward_configs["shaping_weight"] == 0.0:
            return rew

        geodesic_distance = self.obj_dist_to_target()

        if geodesic_distance == -1.0:
            geodesic_distance = self.obj_last_geodesic_distance
        if (
                self.obj_last_geodesic_distance > -0.5 and geodesic_distance > -0.5
        ):  # (robothor limits)
            rew += self.obj_last_geodesic_distance - geodesic_distance
        self.obj_last_geodesic_distance = geodesic_distance

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if self._took_end_action:
            if self._success is not None:
                reward += (
                    self.reward_configs["goal_success_reward"]
                    if self._success
                    else self.reward_configs["failed_stop_reward"]
                )

        self._rewards.append(float(reward))
        return float(reward)

    def spl(self):
        if not self._success:
            return 0.0
        li = self.optimal_distance
        pi = self.num_moves_made * self.env._grid_size
        res = li / (max(pi, li) + 1e-8)
        return res

    def dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            dis.append(self.env.object_distance_to_point(id, self.task_info["target"]))
        id = idx[np.argmin(dis)]
        return self.env.distance_to_point(self.env.all_objects()[id]["position"])

    def obj_dist_to_target(self):
        objs, idx = self.env.get_objects_and_idx_by_type(self.task_info["object_type"])
        dis = []
        for id in idx:
            dis.append(self.env.object_distance_to_point(id, self.task_info["target"]))
        return min(dis)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        total_reward = float(np.sum(self._rewards))
        self._rewards = []

        if self._success is None:
            return {}

        dist2tget = self.obj_dist_to_target()
        spl = self.spl()

        return {
            **super(ObjectPlacementTask, self).metrics(),
            "success": self._success,  # False also if no path to target
            "total_reward": total_reward,
            "dist_to_target": dist2tget,
            "spl": spl,
            "both_in_reachable_points": self.last_both_in_path,
        }

    def query_expert(self, end_action_only: bool = False, **kwargs) -> Tuple[int, bool]:
        if self._is_goal_in_range():
            return self.class_action_names().index(END), True
        if end_action_only:
            return 0, False
        else:
            raise NotImplementedError