import torch
from torch.distributions import Distribution
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
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
from torch.distributions import Normal

import abc
class ADMM(abc.ABC):
    """ADMM

    Args:

    """
    _p_dist: Normal

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self._lambda: float = lagrangian_multiplier_init
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self.update_steps: int = 0
        self._lagrangian_multiplier_return = 0
        self.Lc: float = 0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        return self._lagrangian_multiplier_return

    def para_update(self, Jc: float) -> None:
        Lc = Jc - self._cost_limit

        # 可以尝试注释下面赋值代码
        self.Lc = Lc

        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = min(-Lc, 0)

        if float(- Lc - self._lambda/self._rho) > 0:
            new_xi = float(- Lc - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))

        self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi

        delta_lambda = float(self._xi + Lc)
        self._lambda_resisual = delta_lambda * self._lambda_lr

        self._lambda = self._lambda + delta_lambda * self._lambda_lr
        self._lambda = min(self._lambda, 10)

    def record_Jc(self, Jc: float):
        self.Lc = Jc - self._cost_limit


@registry.register
class F2CPO(PolicyGradient):
    """

    """
    def _init(self) -> None:
        """

        """
        super()._init()
        self._admm: ADMM = ADMM(**self._cfgs.admm_cfgs)
        self.clip_adv = True
        self.clip_adv_c = False

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
        self._logger.register_key('Value/Adv_c')
        self._logger.register_key('Loss/Loss_pi_cost')
        self._logger.register_key('Loss/Loss_proximal')
        self._logger.register_key('Train/Weight')

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute pi/actor loss.

        In FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
                L = \nabla_{\theta} D_{K L} \left( \pi_{\theta}^{'} \| \pi_{\theta} \right)[s]
                - \frac{1}{\eta} \underset{a \sim \pi_{\theta}}{\mathbb{E}} \left[
                    \frac{\nabla_{\theta} \pi_{\theta} (a \mid s)}{\pi_{\theta}(a \mid s)}
                    \left( A^{R}_{\pi_{\theta}} (s, a) - \lambda A^C_{\pi_{\theta}} (s, a) \right)
                \right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta}` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` sampled from buffer.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)
        loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * (
            kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        ).type(torch.float32)
        loss = loss.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Train/PolicyStd': std,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        FOCOPS uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        return (adv_r - self._admm.lagrangian_multiplier * adv_c) / (
            1 + self._admm.lagrangian_multiplier
        )


    def _adv_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        if self.clip_adv_c:
            ratio_cliped = torch.clamp(
                ratio,
                1 - self._cfgs.algo_cfgs.clip,
                1 + self._cfgs.algo_cfgs.clip,
            )
            surr_cadv = torch.max(ratio * adv_c, ratio_cliped * adv_c).mean()
        else:
            surr_cadv = (ratio * adv_c).mean()

        self._logger.store(
            {
                'Loss/Loss_pi_cost': surr_cadv.mean().item(),
            },
        )

        return surr_cadv.mean()


    def _update(self) -> None:
        # note that logger already uses MPI statistics across all processes.
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._admm.record_Jc(Jc)

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
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        dataloader = DataLoader(
            dataset=TensorDataset(
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
                old_mean,
                old_std,
            ),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for j in range(self._cfgs.algo_cfgs.inner_iters):
                for (
                    obs_batch,
                    act_batch,
                    logp_batch,
                    target_value_r_batch,
                    target_value_c_batch,
                    adv_r_batch,
                    adv_c_batch,
                    old_mean,
                    old_std,
                ) in dataloader:
                    self._update_reward_critic(obs_batch, target_value_r_batch)
                    if self._cfgs.algo_cfgs.use_cost:
                        self._update_cost_critic(obs_batch, target_value_c_batch)

                    self._p_dist = Normal(old_mean, old_std)
                    self._update_actor(obs_batch, act_batch, logp_batch, adv_r_batch, adv_c_batch)

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

            Jc_e = self._adv_cost(obs, act, logp, adv_c).detach() + Jc
            self._admm.para_update(Jc_e)
            self._logger.store(
                {
                    'Train/Weight': self._admm._rho*(Jc_e+self._admm._xi-self._admm._cost_limit) + self._admm._lambda,  # pylint: disable=undefined-loop-variable
                },
            )

        self._logger.store(
            {
                'Train/StopIter': update_counts,  # pylint: disable=undefined-loop-variable
                'Value/Adv': adv_r.mean().item(),
                'Value/Adv_c': adv_c.mean().item(),
                'Train/KL': final_kl,
            },
        )

        self._logger.store({'Metrics/LagrangeMultiplier': self._admm._lambda})
        self._logger.store({'Metrics/Sigma': self._admm.sigma})
        self._logger.store({'Metrics/Xi': self._admm._xi})

    # def _update_actor(  # pylint: disable=too-many-arguments
    #     self,
    #     obs: torch.Tensor,
    #     act: torch.Tensor,
    #     logp: torch.Tensor,
    #     adv_r: torch.Tensor,
    #     adv_c: torch.Tensor,
    # ) -> None:
    #     """Update policy network under a double for loop.

    #     #. Compute the loss function.
    #     #. Clip the gradient if ``use_max_grad_norm`` is ``True``.
    #     #. Update the network by loss function.

    #     .. warning::
    #         For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
    #         the ``KL divergence`` between the old policy and the new policy is calculated.
    #         And the ``KL divergence`` is used to determine whether the update is successful.
    #         If the ``KL divergence`` is too large, the update will be terminated.

    #     Args:
    #         obs (torch.Tensor): The ``observation`` sampled from buffer.
    #         act (torch.Tensor): The ``action`` sampled from buffer.
    #         logp (torch.Tensor): The ``log_p`` sampled from buffer.
    #         adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
    #         adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.
    #     """
    #     loss = self._loss_pi(obs, act, logp, adv_r) + self.proximal_loss(obs, act, logp, adv_c)
    #     self._actor_critic.actor_optimizer.zero_grad()
    #     loss.backward()
    #     if self._cfgs.algo_cfgs.use_max_grad_norm:
    #         clip_grad_norm_(
    #             self._actor_critic.actor.parameters(),
    #             self._cfgs.algo_cfgs.max_grad_norm,
    #         )
    #     distributed.avg_grads(self._actor_critic.actor)
    #     self._actor_critic.actor_optimizer.step()