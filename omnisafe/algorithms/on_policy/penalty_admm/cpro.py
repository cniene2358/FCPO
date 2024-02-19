import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.admm import ADMM3


@registry.register
class CPRO(PPO):
    def _init(self) -> None:
        super()._init()
        self._admm: ADMM3 = ADMM3(**self._cfgs.admm_cfgs)

    def _init_log(self) -> None:
        """Log the ACPO specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Metrics/LagrangeMultiplier | The PID-Lagrange multiplier. |
        +----------------------------+------------------------------+
        +============================+==============================+
        | Metrics/Sigma              | The penalty parameter.       |
        +----------------------------+------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Metrics/Sigma')
        self._logger.register_key('Metrics/Xi')

    def _update(self) -> None:

        Jc = self._logger.get_stats('Metrics/EpCost')[0]

        # update the policy and value function
        super()._update()

        # note that logger already uses MPI statistics across all processes.
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        Lc = self._loss_pi_cost(obs, act, logp, adv_c).detach() + Jc
        
        # update Lagrange multiplier parameter
        self._admm.para_update(Lc)
        
        self._logger.store({'Metrics/LagrangeMultiplier': self._admm._lambda})
        self._logger.store({'Metrics/Sigma': self._admm.sigma})
        self._logger.store({'Metrics/Xi': self._admm._xi})

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        surr_cadv = (ratio * adv_c).mean()
        loss_cost = surr_cadv - self._cfgs.admm_cfgs.cost_limit
        return loss_cost.mean()

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        penalty = self._admm.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
