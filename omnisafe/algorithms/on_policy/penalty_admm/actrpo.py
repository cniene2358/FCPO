import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.admm import ADMM, ADMM2, ADMM3


@registry.register
class ACTRPO(TRPO):
    """

    """

    def _init(self) -> None:
        """

        """
        super()._init()
        self._admm: ADMM2 = ADMM2(**self._cfgs.admm_cfgs)
        # self._admm: ADMM2 = ADMM2(**self._cfgs.admm_cfgs)
        # self._admm: ADMM3 = ADMM3(**self._cfgs.admm_cfgs)

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
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        super()._update()
        self._admm.para_update(Jc)
        # then update the policy and value function


        self._logger.store({'Metrics/LagrangeMultiplier': self._admm._lambda})
        self._logger.store({'Metrics/Sigma': self._admm.sigma})
        self._logger.store({'Metrics/Xi': self._admm._xi})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        CPPOPID uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """

        penalty = self._admm.lagrangian_multiplier
        # penalty = 100
        return (adv_r - penalty * adv_c) / (1 + penalty)
