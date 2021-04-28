"""Defining imitation losses for actor critic type models."""

from typing import Dict, cast

import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import AbstractActorCriticLoss
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.misc import ActorCriticOutput
from allenact.base_abstractions.distributions import CategoricalDistr

class YesNoImitation(AbstractActorCriticLoss):
    def __init__(self, yes_action_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yes_action_index = yes_action_index

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        if isinstance(actor_critic_output, dict):
            actor_critic_output = actor_critic_output["ac_output"]
        observations = cast(Dict[str, torch.Tensor], batch["observations"])

        assert "expert_action" in observations

        expert_actions_and_mask = observations["expert_action"]
        if len(expert_actions_and_mask.shape) == 3:
            # No agent dimension in expert action
            expert_actions_and_mask = expert_actions_and_mask.unsqueeze(-2)

        assert expert_actions_and_mask.shape[-1] == 2
        expert_actions_and_mask_reshaped = expert_actions_and_mask.view(-1, 2)

        expert_actions = expert_actions_and_mask_reshaped[:, 0].view(
            *expert_actions_and_mask.shape[:-1]
        )
        expert_actions_masks = (
            expert_actions_and_mask_reshaped[:, 1]
            .float()
            .view(*expert_actions_and_mask.shape[:-1])
        )

        log_probs_yes_action = actor_critic_output.distributions.log_prob(
            cast(
                torch.LongTensor,
                self.yes_action_index + torch.zeros_like(expert_actions),
            )
        )
        log_probs_not_yes_action = torch.log(1.0 - torch.exp(log_probs_yes_action))  # type: ignore
        expert_action_was_yes_action = (
            expert_actions_masks * (expert_actions == self.yes_action_index).float()
        )

        total_loss = -(
            (
                log_probs_yes_action * expert_action_was_yes_action
                + log_probs_not_yes_action * (1.0 - expert_action_was_yes_action)  # type: ignore
            )
        ).mean()

        return (total_loss, {"yes_action_cross_entropy": total_loss.item(),})
