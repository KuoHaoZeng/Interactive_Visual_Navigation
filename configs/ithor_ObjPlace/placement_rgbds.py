import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig, PPO
from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact.utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay

from interactive_navigation.tasks import ObjectPlacementTask
from interactive_navigation.models import PlacementRGBDSActorCriticSimpleConvRNN
from interactive_navigation.constants import END
from interactive_navigation.losses import YesNoImitation
from interactive_navigation.sensors import (
    RGBSensorThor,
    DepthSensorIThor,
    GoalObjectTypeThorSensor,
    GPSCompassSensorIThor,
    ClassSegmentationSensorThor,
)
from configs.ithor_ObjPlace.placement_base import PlacementBaseConfig


class PlacementRGBDSConfig(PlacementBaseConfig):
    def __init__(self):
        super().__init__()

        self.SENSORS = [
            RGBSensorThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid="rgb",
            ),
            DepthSensorIThor(
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                use_normalization=True,
                uuid="depth",
            ),
            GoalObjectTypeThorSensor(self.OBSTACLES_TYPES),
            GPSCompassSensorIThor(),
            ClassSegmentationSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                height=self.SCREEN_SIZE,
                width=self.SCREEN_SIZE,
                uuid="seg"
            ),
            ExpertActionSensor(
                nactions=len(ObjectPlacementTask.class_action_names()),
                uuid="expert_action",
                expert_args={"end_action_only": True}
            ),
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "goal_object_type_ind",
            "target_coordinates_ind",
            "seg",
            "expert_action",
        ]

    @classmethod
    def tag(cls):
        return "Placement-RGBDS"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(10000000)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 1000000
        log_interval = 100
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": PPO(**PPOConfig),
                          "yn_im_loss": YesNoImitation(yes_action_index=ObjectPlacementTask.class_action_names().index(END)),},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss", "yn_im_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return PlacementRGBDSActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(ObjectPlacementTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            object_sensor_uuid="goal_object_type_ind",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            object_type_embedding_dim=32,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
