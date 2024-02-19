import torch
from torch.distributions import Distribution
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.admm import ADMM, ADMM2, ADMM3
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset



@registry.register
class CPPO(PPO):
    """

    """
    def _init(self) -> None:
        """

        """
        super()._init()
        self._admm: ADMM3 = ADMM3(**self._cfgs.admm_cfgs)
        self.gamma = 0.99

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
        # self._logger.register_key('Metrics/LagrangeMultiplier_return')

    def _update(self) -> None:
        # note that logger already uses MPI statistics across all processes.
        Jc = self._logger.get_stats('Metrics/EpCost')[0]

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

        original_obs = obs
        old_distribution = self._actor_critic.actor(obs)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)
                self._update_actor(obs, act, logp, adv_r, adv_c)
            new_distribution = self._actor_critic.actor(original_obs)
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            kl = distributed.dist_avg(kl)
            final_kl = kl
            update_counts += 1
            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break
            Lc = self._loss_pi_cost(obs, act, logp, adv_c).detach() + Jc
            self._admm.para_update(Lc)


        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': final_kl,
            },
        )

        self._logger.store({'Metrics/LagrangeMultiplier': self._admm._lambda})
        self._logger.store({'Metrics/Sigma': self._admm.sigma})
        self._logger.store({'Metrics/Xi': self._admm._xi})
        # self._logger.store({'Metrics/LagrangeMultiplier_return': self._admm._lagrangian_multiplier_return})


    def proximal_loss(self, obs, act, logp, adv_c):
        Lc = self._loss_pi_cost(obs, act, logp, adv_c) + self._logger.get_stats('Metrics/EpCost')[0]
        loss = self._admm._rho/2 * (Lc + self._admm._xi + self._admm._lambda/self._admm._rho)**2
        return loss

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

        # --------------- v1 --------------
        surr_cadv = (ratio * adv_c).mean()
        # ---------------- v2 ---------------
        # ratio_cliped = torch.clamp(
        #     ratio,
        #     1 - self._cfgs.algo_cfgs.clip,
        #     1 + self._cfgs.algo_cfgs.clip,
        # )
        # surr_cadv = torch.min(ratio * adv_c, ratio_cliped * adv_c).mean()
        #---------------------------------------

        loss_cost = surr_cadv - self._cfgs.admm_cfgs.cost_limit
        # loss_cost = (1-self.gamma)**(-1) * surr_cadv - self._cfgs.admm_cfgs.cost_limit
        return loss_cost.mean()

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network under a double for loop.

        #. Compute the loss function.
        #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
        #. Update the network by loss function.

        .. warning::
            For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
            the ``KL divergence`` between the old policy and the new policy is calculated.
            And the ``KL divergence`` is used to determine whether the update is successful.
            If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log_p`` sampled from buffer.
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
        """
        loss = self._loss_pi(obs, act, logp, adv_r) + self.proximal_loss(obs, act, logp, adv_c)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.actor)
        self._actor_critic.actor_optimizer.step()