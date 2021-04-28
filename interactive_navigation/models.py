import typing, gym, torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Union, List, cast, Sequence
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.model_utils import make_cnn, compute_cnn_output

class RGBDSCNN(nn.Module):
    def __init__(
            self,
            observation_space: SpaceDict,
            output_size: int,
            layer_channels: Sequence[int] = (32, 64, 32),
            kernel_sizes: Sequence[Tuple[int, int]] = ((8, 8), (4, 4), (3, 3)),
            layers_stride: Sequence[Tuple[int, int]] = ((4, 4), (2, 2), (1, 1)),
            paddings: Sequence[Tuple[int, int]] = ((0, 0), (0, 0), (0, 0)),
            dilations: Sequence[Tuple[int, int]] = ((1, 1), (1, 1), (1, 1)),
            rgb_uuid: str = "rgb",
            depth_uuid: str = "depth",
            seg_uuid: str = "seg",
            flatten: bool = True,
            output_relu: bool = True,
    ):
        super().__init__()

        self.rgb_uuid = rgb_uuid
        if self.rgb_uuid in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces[self.rgb_uuid].shape[2]
            assert self._n_input_rgb >= 0
        else:
            self._n_input_rgb = 0

        self.depth_uuid = depth_uuid
        if self.depth_uuid in observation_space.spaces:
            self._n_input_depth = observation_space.spaces[self.depth_uuid].shape[2]
            assert self._n_input_depth >= 0
        else:
            self._n_input_depth = 0

        self.seg_uuid = seg_uuid
        self._n_input_seg = 1

        if not self.is_blind:
            # hyperparameters for layers
            self._cnn_layers_channels = list(layer_channels)
            self._cnn_layers_kernel_size = list(kernel_sizes)
            self._cnn_layers_stride = list(layers_stride)
            self._cnn_layers_paddings = list(paddings)
            self._cnn_layers_dilations = list(dilations)

            if self._n_input_rgb > 0:
                input_rgb_cnn_dims = np.array(
                    observation_space.spaces[self.rgb_uuid].shape[:2], dtype=np.float32
                )
                self.rgb_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_rgb_cnn_dims,
                    input_channels=self._n_input_rgb,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_depth > 0:
                input_depth_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.depth_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_depth_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

            if self._n_input_seg > 0:
                input_seg_cnn_dims = np.array(
                    observation_space.spaces[self.depth_uuid].shape[:2],
                    dtype=np.float32,
                )
                self.seg_cnn = self.make_cnn_from_params(
                    output_size=output_size,
                    input_dims=input_seg_cnn_dims,
                    input_channels=self._n_input_depth,
                    flatten=flatten,
                    output_relu=output_relu,
                )

    def make_cnn_from_params(
            self,
            output_size: int,
            input_dims: np.ndarray,
            input_channels: int,
            flatten: bool,
            output_relu: bool,
    ) -> nn.Module:
        output_dims = input_dims
        for kernel_size, stride, padding, dilation in zip(
                self._cnn_layers_kernel_size,
                self._cnn_layers_stride,
                self._cnn_layers_paddings,
                self._cnn_layers_dilations,
        ):
            # noinspection PyUnboundLocalVariable
            output_dims = self._conv_output_dim(
                dimension=output_dims,
                padding=np.array(padding, dtype=np.float32),
                dilation=np.array(dilation, dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        # noinspection PyUnboundLocalVariable
        cnn = make_cnn(
            input_channels=input_channels,
            layer_channels=self._cnn_layers_channels,
            kernel_sizes=self._cnn_layers_kernel_size,
            strides=self._cnn_layers_stride,
            paddings=self._cnn_layers_paddings,
            dilations=self._cnn_layers_dilations,
            output_height=output_dims[0],
            output_width=output_dims[1],
            output_channels=output_size,
            flatten=flatten,
            output_relu=output_relu,
        )
        self.layer_init(cnn)

        return cnn

    @staticmethod
    def _conv_output_dim(
            dimension: Sequence[int],
            padding: Sequence[int],
            dilation: Sequence[int],
            kernel_size: Sequence[int],
            stride: Sequence[int],
    ) -> Tuple[int, ...]:
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                                (
                                        dimension[i]
                                        + 2 * padding[i]
                                        - dilation[i] * (kernel_size[i] - 1)
                                        - 1
                                )
                                / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    @staticmethod
    def layer_init(cnn) -> None:
        for layer in cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        if self.is_blind:
            return None

        def check_use_agent(new_setting):
            if use_agent is not None:
                assert (
                        use_agent is new_setting
                ), "rgb and depth must both use an agent dim or none"
            return new_setting

        cnn_output_list: List[torch.Tensor] = []
        use_agent: Optional[bool] = None

        if self._n_input_rgb > 0:
            use_agent = check_use_agent(len(observations[self.rgb_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.rgb_cnn, observations[self.rgb_uuid])
            )

        if self._n_input_depth > 0:
            use_agent = check_use_agent(len(observations[self.depth_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.depth_cnn, observations[self.depth_uuid])
            )

        if self._n_input_seg > 0:
            use_agent = check_use_agent(len(observations[self.seg_uuid]) == 6)
            cnn_output_list.append(
                compute_cnn_output(self.seg_cnn, observations[self.seg_uuid])
            )

        if use_agent:
            channels_dim = 3  # [step, sampler, agent, channel (, height, width)]
        else:
            channels_dim = 2  # [step, sampler, channel (, height, width)]

        return torch.cat(cnn_output_list, dim=channels_dim)


class ObstaclesNavRGBDActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        embed_coordinates=False,
        coordinate_embedding_dim=8,
        coordinate_dims=2,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ObstaclesNavRGBDSActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 3, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = RGBDSCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ObstaclesNavRGBDKNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x),
            values=self.critic(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class ObstaclesNavRGBDKvNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x),
            values=self.critic(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementRGBDActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementRGBDSActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 3, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = RGBDSCNN(observation_space, hidden_size)

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
            current_actions: torch.FloatTensor = None,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementRGBDKNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)

        hidden_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x),
            values=self.critic(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)


class PlacementRGBDKvNIEActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            object_sensor_uuid: str,
            obstacle_keypoints_sensor_uuid: str,
            hidden_size=512,
            embed_coordinates=False,
            coordinate_embedding_dim=8,
            coordinate_dims=2,
            object_type_embedding_dim=8,
            obstacle_type_embedding_dim=8,
            obstacle_state_hidden_dim=16,
            num_obstacle_types=20,
            num_rnn_layers=1,
            rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)

        self.goal_sensor_uuid = goal_sensor_uuid
        self.object_sensor_uuid = object_sensor_uuid
        self._hidden_size = hidden_size
        self.embed_coordinates = embed_coordinates
        if self.embed_coordinates:
            self.coorinate_embedding_size = coordinate_embedding_dim
        else:
            self.coorinate_embedding_size = coordinate_dims
        self._n_object_types = self.observation_space.spaces[self.object_sensor_uuid].n
        self.object_type_embedding_size = object_type_embedding_dim
        self.obstacle_keypoints_sensor_uuid = obstacle_keypoints_sensor_uuid
        self.obstacle_type_embedding_size = obstacle_type_embedding_dim
        self.obstacle_state_hidden_dim = obstacle_state_hidden_dim

        self.sensor_fusion = False
        if "rgb" in observation_space.spaces and "depth" in observation_space.spaces:
            self.sensor_fuser = nn.Linear(hidden_size * 2, hidden_size)
            self.sensor_fusion = True

        self.visual_encoder = SimpleCNN(observation_space, hidden_size, "rgb", "depth")

        self.state_encoder = RNNStateEncoder(
            (0 if self.is_blind else self.recurrent_hidden_state_size)
            + self.object_type_embedding_size + self.coorinate_embedding_size + obstacle_state_hidden_dim * action_space.n,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
            )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        if self.embed_coordinates:
            self.coordinate_embedding = nn.Linear(
                coordinate_dims, coordinate_embedding_dim
            )

        self.object_type_embedding = nn.Embedding(
            num_embeddings=self._n_object_types,
            embedding_dim=object_type_embedding_dim,
        )

        # Action embedding
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space.n, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.num_actions = self.action_space.n

        # Object hidden state encoding
        self.meta_embedding = nn.Embedding(
            num_embeddings=num_obstacle_types, embedding_dim=self.obstacle_state_hidden_dim
        )
        self.rotation_encoding = nn.Sequential(
            nn.Linear(24, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )
        self.state_encoding = nn.Sequential(
            nn.Linear(3, self.obstacle_state_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim // 2, self.obstacle_state_hidden_dim),
        )

        # NIE
        self.NIE = nn.Sequential(
            nn.Linear(hidden_size + self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 12)
        )
        self.NIE[4].weight.data.zero_()
        self.NIE[4].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # NIE attention
        self.NIE_atten = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim, 1)
        )

        # NIE Summary
        self.NIE_summary = nn.Sequential(
            nn.Linear(self.obstacle_state_hidden_dim * 4, self.obstacle_state_hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 3, self.obstacle_state_hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(self.obstacle_state_hidden_dim * 2, self.obstacle_state_hidden_dim),
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def get_target_coordinates_encoding(self, observations):
        if self.embed_coordinates:
            return self.coordinate_embedding(
                observations[self.goal_sensor_uuid].to(torch.float32)
            )
        else:
            return observations[self.goal_sensor_uuid].to(torch.float32)

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        # noinspection PyTypeChecker
        return self.object_type_embedding(  # type:ignore
            observations[self.object_sensor_uuid].to(torch.int64)
        )

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        target_encoding = self.get_target_coordinates_encoding(observations)
        object_encoding = self.get_object_type_encoding(
            cast(Dict[str, torch.FloatTensor], observations)
        )
        x: Union[torch.Tensor, List[torch.Tensor]]
        x = [target_encoding, object_encoding]

        if not self.is_blind:
            perception_embed = self.visual_encoder(observations)
            if self.sensor_fusion:
                perception_embed = self.sensor_fuser(perception_embed)
            x = [perception_embed] + x

        nb, ng, no, np, nd = observations[self.obstacle_keypoints_sensor_uuid].shape
        nh = self.obstacle_state_hidden_dim

        keypoints = observations[self.obstacle_keypoints_sensor_uuid].view(nb, ng, no, np, nd)
        obstacles_index = torch.arange(0, no).to(target_encoding.device).long()
        obstacles_meta_hidden = self.meta_embedding(obstacles_index)
        obstacles_rot_hidden = self.rotation_encoding(keypoints.view(nb, ng, no, np*nd))
        obstacles_state_hidden = self.state_encoding(keypoints.mean(3))

        na = self.num_actions
        actions_index = torch.arange(0, na).to(target_encoding.device).long()
        a_feature = self.action_embedding(actions_index).view(-1, na, nh)

        keypoints = keypoints.view(nb, ng, no, 1, np, nd).repeat(1, 1, 1, na, 1, 1)
        keypoints_homo = torch.cat((keypoints, torch.ones(nb, ng, no, na, np, 1).to(target_encoding.device)), 5)
        obstacles_meta_hidden = obstacles_meta_hidden.view(1, 1, no, 1, nh).repeat(nb, ng, 1, na, 1)
        obstacles_rot_hidden = obstacles_rot_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        obstacles_state_hidden = obstacles_state_hidden.view(nb, ng, no, 1, nh).repeat(1, 1, 1, na, 1)
        a_feature = a_feature.view(1, 1, 1, na, nh).repeat(nb, ng, no, 1, 1)
        perception_embed_hidden = perception_embed.view(nb, ng, 1, 1, self._hidden_size).repeat(1, 1, no, na, 1)

        hidden_feature = torch.cat((perception_embed_hidden, obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        NIE_hidden = self.NIE(hidden_feature)
        NIE_hidden = NIE_hidden
        M = NIE_hidden.view(nb, ng, no, na, 3, 4)
        M_test = M.clone()
        M_test[:, :, :, 7] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        M_test[:, :, :, 5] = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(M.device)
        new_keypoints = torch.matmul(M, keypoints_homo.transpose(4, 5)).transpose(4, 5)
        new_obstacles_state_hidden = self.state_encoding(new_keypoints.mean(4))

        atten_feature = torch.cat((obstacles_rot_hidden, obstacles_meta_hidden, a_feature), dim=4)
        hidden_feature = torch.cat((obstacles_meta_hidden, obstacles_state_hidden, new_obstacles_state_hidden,
                                    a_feature), dim=4)
        NIE_atten_score = self.NIE_atten(atten_feature)
        NIE_atten_prob = nn.functional.softmax(NIE_atten_score, 2)
        NIE_atten_hidden = (hidden_feature * NIE_atten_prob).sum(2)
        NIE_atten_hidden = self.NIE_summary(NIE_atten_hidden)
        NIE_atten_hidden = NIE_atten_hidden.view(nb, ng, -1)
        x.append(NIE_atten_hidden)

        x = torch.cat(x, dim=-1)
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        out = ActorCriticOutput(
            distributions=self.actor(x),
            values=self.critic(x),
            extras={"nie_output": new_keypoints}
        )

        return out, memory.set_tensor("rnn", rnn_hidden_states)

