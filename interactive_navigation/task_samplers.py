import random, gzip, json, gym
from typing import List, Dict, Optional, Any, Union, Tuple

from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.cache_utils import str_to_pos_for_cache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed

from interactive_navigation.environment import IThorEnvironment
from interactive_navigation.tasks import ObstaclesNavTask, ObjectPlacementTask

class ObstaclesNavDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: self.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ObstaclesNavTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ObstaclesNavTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObstaclesNavTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"], rotation=episode["initial_orientation"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not self.env.spawn_obj(obj):
                return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        self._last_sampled_task = ObstaclesNavTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

class ObjectPlacementDatasetTaskSampler(TaskSampler):
    def __init__(
            self,
            scenes: List[str],
            scene_directory: str,
            sensors: List[Sensor],
            max_steps: int,
            env_args: Dict[str, Any],
            action_space: gym.Space,
            rewards_config: Dict,
            seed: Optional[int] = None,
            deterministic_cudnn: bool = False,
            loop_dataset: bool = True,
            shuffle_dataset: bool = True,
            allow_flipping=False,
            env_class=IThorEnvironment,
            **kwargs,
    ) -> None:
        self.rewards_config = rewards_config
        self.env_args = env_args
        self.scenes = scenes
        self.shuffle_dataset: bool = shuffle_dataset
        self.episodes = {
            scene: self.load_dataset(
                scene, scene_directory + "/episodes"
            )
            for scene in scenes
        }
        self.env_class = env_class
        self.env: Optional[IThorEnvironment] = None
        self.sensors = sensors
        self.max_steps = max_steps
        self._action_space = action_space
        self.allow_flipping = allow_flipping
        self.scene_counter: Optional[int] = None
        self.scene_order: Optional[List[str]] = None
        self.scene_id: Optional[int] = None
        # get the total number of tasks assigned to this process
        if loop_dataset:
            self.max_tasks = None
        else:
            self.max_tasks = sum(len(self.episodes[scene]) for scene in self.episodes)
        self.reset_tasks = self.max_tasks
        self.scene_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ObjectPlacementTask] = None

        self.seed: Optional[int] = None
        self.set_seed(seed)

        if deterministic_cudnn:
            set_deterministic_cudnn()

        self.reset()

    def _create_environment(self) -> IThorEnvironment:
        env = self.env_class(**self.env_args)
        return env

    @staticmethod
    def load_dataset(scene: str, base_directory: str) -> List[Dict]:
        filename = (
            "/".join([base_directory, scene])
            if base_directory[-1] != "/"
            else "".join([base_directory, scene])
        )
        filename += ".json.gz"
        fin = gzip.GzipFile(filename, "r")
        json_bytes = fin.read()
        fin.close()
        json_str = json_bytes.decode("utf-8")
        data = json.loads(json_str)
        random.shuffle(data)
        return data

    @property
    def __len__(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ObjectPlacementTask]:
        return self._last_sampled_task

    def close(self) -> None:
        if self.env is not None:
            self.env.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.

        # Returns

        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def next_task(self, force_advance_scene: bool = False) -> Optional[ObjectPlacementTask]:
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        if self.episode_index >= len(self.episodes[self.scenes[self.scene_index]]):
            self.scene_index = (self.scene_index + 1) % len(self.scenes)
            # shuffle the new list of episodes to train on
            if self.shuffle_dataset:
                random.shuffle(self.episodes[self.scenes[self.scene_index]])
            self.episode_index = 0

        scene = self.scenes[self.scene_index]
        episode = self.episodes[scene][self.episode_index]
        if self.env is not None:
            #if scene.replace("_physics", "") != self.env.scene_name.replace(
            #        "_physics", ""
            #):
            self.env.reset(scene_name=scene, filtered_objects=[])
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene, filtered_objects=[])

        def to_pos(s):
            if isinstance(s, (Dict, Tuple)):
                return s
            if isinstance(s, float):
                return {"x": 0, "y": s, "z": 0}
            return str_to_pos_for_cache(s)

        for k in ["initial_position", "initial_orientation", "target_position"]:
            episode[k] = to_pos(episode[k])

        self.episode_index += 1
        if self.max_tasks is not None:
            self.max_tasks -= 1

        if not self.env.teleport(
                pose=episode["initial_position"],
                rotation=episode["initial_orientation"],
                horizon=episode["initial_horizon"]
        ):
            return self.next_task()

        for obj in episode["spawn_objects"]:
            if not "TargetCircle" in obj["objectType"]:
                if not self.env.spawn_obj(obj):
                    return self.next_task()
            else:
                if not self.env.spawn_target_circle(obj["random_seed"]):
                    return self.next_task()

        self.env.initialize(0.25, renderObjectImage=True, renderDepthImage=True)

        task_info = {
            "scene": scene,
            "initial_position": episode["initial_position"],
            "initial_orientation": episode["initial_orientation"],
            "target": episode["target_position"],
            "shortest_path": episode["shortest_path"],
            "distance_to_target": episode["shortest_path_length"],
            "id": episode["id"],
            "spawn_objects": episode["spawn_objects"],
            "obstacles_types": episode["obstacles_types"],
            "target_type": "TargetCircle",
            "object_type": episode["spawn_objects"][1]["objectType"],
        }

        if self.allow_flipping and random.random() > 0.5:
            task_info["mirrored"] = True
        else:
            task_info["mirrored"] = False

        self._last_sampled_task = ObjectPlacementTask(
            env=self.env,
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )

        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.scene_index = 0
        self.max_tasks = self.reset_tasks

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    @property
    def length(self) -> Union[int, float]:
        """Length.

        # Returns

        Number of total tasks remaining that can be sampled.
        Can be float('inf').
        """
        return float("inf") if self.max_tasks is None else self.max_tasks
