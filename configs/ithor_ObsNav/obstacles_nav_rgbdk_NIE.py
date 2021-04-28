import gym
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig, PPO
from allenact.utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay

from interactive_navigation.tasks import ObstaclesNavTask
from interactive_navigation.models import ObstaclesNavRGBDKNIEActorCriticSimpleConvRNN
from interactive_navigation.losses import NIE_Reg
from interactive_navigation.sensors import (
    RGBSensorThor,
    DepthSensorIThor,
    GPSCompassSensorIThor,
    LocalKeyPoints3DSensorThor,
    GlobalKeyPoints3DSensorThor,
    GlobalObjPoseSensorThor,
    GlobalAgentPoseSensorThor,
    GlobalObjUpdateMaskSensorThor,
    GlobalObjActionMaskSensorThor,
)
from configs.ithor_ObsNav.obstacles_nav_base import ObstaclesNavBaseConfig


class ObstaclesNavRGBDKNIEConfig(ObstaclesNavBaseConfig):
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
            GPSCompassSensorIThor(),
            LocalKeyPoints3DSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="3Dkeypoints_local"
            ),
            GlobalKeyPoints3DSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="3Dkeypoints_global"
            ),
            GlobalObjPoseSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_pose_global"
            ),
            GlobalAgentPoseSensorThor(
                uuid="agent_pose_global"
            ),
            GlobalObjUpdateMaskSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_update_mask"
            ),
            GlobalObjActionMaskSensorThor(
                objectTypes=self.OBSTACLES_TYPES,
                uuid="object_action_mask"
            )
        ]

        self.PREPROCESSORS = []

        self.OBSERVATIONS = [
            "rgb",
            "depth",
            "target_coordinates_ind",
            "3Dkeypoints_local",
            "3Dkeypoints_global",
            "object_pose_global",
            "agent_pose_global",
            "object_update_mask",
            "object_action_mask",
        ]

    @classmethod
    def tag(cls):
        return "ObstaclesNav-RGBDK-NIE"

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
            named_losses={
                "ppo_loss": PPO(**PPOConfig),
                "nie_loss": NIE_Reg(agent_pose_uuid="agent_pose_global",
                                    pose_uuid="object_pose_global",
                                    local_keypoints_uuid="3Dkeypoints_local",
                                    global_keypoints_uuid="3Dkeypoints_global",
                                    obj_update_mask_uuid="object_update_mask",
                                    obj_action_mask_uuid="object_action_mask",)
            },
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss", "nie_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ObstaclesNavRGBDKNIEActorCriticSimpleConvRNN(
            action_space=gym.spaces.Discrete(len(ObstaclesNavTask.class_action_names())),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            goal_sensor_uuid="target_coordinates_ind",
            obstacle_keypoints_sensor_uuid="3Dkeypoints_local",
            hidden_size=512,
            embed_coordinates=False,
            coordinate_dims=2,
            obstacle_type_embedding_dim=32,
            obstacle_state_hidden_dim=64,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
        )
