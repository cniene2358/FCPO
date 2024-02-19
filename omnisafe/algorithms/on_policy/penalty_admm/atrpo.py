import torch
from torch.distributions import Distribution
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.admm import ADMM, ADMM2, ADMM3
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class ATRPO(TRPO):
    """

    """
    def _init(self) -> None:
        """

        """
        super()._init()
        self._admm: ADMM2 = ADMM2(**self._cfgs.admm_cfgs)

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
        # note that logger already uses MPI statistics across all processes.
        # Jc = + self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        for i in range(self._cfgs.algo_cfgs.policy_update_iters):
            super()._update()
        self._admm.para_update(Jc)
        # then update the policy and value function


        self._logger.store({'Metrics/LagrangeMultiplier': self._admm._lambda})
        self._logger.store({'Metrics/Sigma': self._admm.sigma})
        self._logger.store({'Metrics/Xi': self._admm._xi})

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        # adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss = self._loss_pi(obs, act, logp, adv_r) + self.proximal_loss(obs, act, logp, adv_c)
        loss_before = distributed.dist_avg(loss).item()
        p_dist = self._actor_critic.actor(obs)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        step_direction, accept_step = self._search_step_size(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_before=loss_before,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss = self._loss_pi(obs, act, logp, adv_r) + self.proximal_loss(obs, act, logp, adv_c)

        self._logger.store(
            {
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
                'Misc/AcceptanceStep': accept_step,
                # 'Metrics/LagrangeMultiplier': self._admm._lambda,
                # 'Metrics/Sigma': self._admm.sigma,
                # 'Metrics/Xi': self._admm._xi,
            },
        )

    def proximal_loss(self, obs, act, logp, adv_c):
        loss = self._admm._rho/2 * (-self._loss_pi(obs, act, logp, adv_c) + self._admm.Jc - self._admm._cost_limit + self._admm._xi + self._admm._lambda/self._admm._rho)**2
        return loss

    # pylint: disable-next=too-many-arguments,too-many-locals,arguments-differ
    def _search_step_size(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        loss_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
    ):
        """TRPO performs `line-search <https://en.wikipedia.org/wiki/Line_search>`_ until constraint satisfaction.

        .. hint::
            TRPO search around for a satisfied step of policy update to improve loss and reward performance. The search
            is done by line-search, which is a way to find a step size that satisfies the constraint. The constraint is
            the KL-divergence between the old policy and the new policy.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.

        Returns:
            The tuple of final update direction and acceptance step size.
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grads.dot(step_direction)

        final_kl = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(total_steps):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(self._actor_critic.actor, new_theta)

            with torch.no_grad():
                loss = self._loss_pi(obs, act, logp, adv_r) + self.proximal_loss(obs, act, logp, adv_c)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
                kl = distributed.dist_avg(kl).mean().item()
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss.item()
            # average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            loss_improve = distributed.dist_avg(loss_improve)
            self._logger.log(f'Expected Improvement: {expected_improve} Actual: {loss_improve}')
            if not torch.isfinite(loss):
                self._logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self._logger.log('INFO: did not improve improve <0')
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                self._logger.log(f'Accept step at i={acceptance_step}')
                final_kl = kl
                break
            step_frac *= decay
        else:
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        set_param_values_to_model(self._actor_critic.actor, theta_old)

        self._logger.store(
            {
                'Train/KL': final_kl,
            },
        )

        return step_frac * step_direction, acceptance_step
