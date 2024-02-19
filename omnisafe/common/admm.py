from __future__ import annotations

import abc
from collections import deque
import torch

# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class ADMM(abc.ABC):  # noqa: B024
    """ADMM

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
        sigma_p: float,
        sigma_d: float,
        rho_steps: int,
        max_sigma: float
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self.d_delay = d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._lambda: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self.d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self._rho_steps: int = rho_steps
        self.update_steps: int = 0
        self._sigma_p: float = sigma_p
        self._sigma_d: float = sigma_d
        self._max_sigma: float = max_sigma
        self._min_xi: float = -1000
        self.Jc: float = 0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        return self._lambda

    def para_update(self, ep_cost_avg: float) -> None:
        self.Jc = ep_cost_avg
        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = min(0, self._cost_limit - ep_cost_avg)
        elif self.update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit > 1.1:
            self.sigma *= self._sigma_p
            self.sigma = min(self._max_sigma, self.sigma)
            self._min_xi = min(0, max(self._min_xi, self._xi))
        if self.update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit < 1.1:
            self.sigma *= self._sigma_d

        if float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho) > 0:
            new_xi = float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - ep_cost_avg + self._cost_limit - self._lambda/self._rho))

        self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi
        # self._xi = max(self._min_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

        delta_lambda = float(self._xi + ep_cost_avg - self._cost_limit)
        if delta_lambda < 0:
            self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr * 2)
        else:
            self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        self._lambda = min(self._lambda, self._penalty_max)

        # self._cost_ds.append(self._cost_d)


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class ADMM2(abc.ABC):  # noqa: B024
    """ADMM

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
        sigma_p: float,
        sigma_d: float,
        rho_steps: int,
        max_sigma: float
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self.d_delay = d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._lambda: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self.d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self._rho_steps: int = rho_steps
        self.update_steps: int = 0
        self._sigma_p: float = sigma_p
        self._sigma_d: float = sigma_d
        self._max_sigma: float = max_sigma
        self._min_xi: float = -1000
        self.Jc: float = 0

        #--------- ADD more attributes-----
        self._lagrangian_multiplier_return = 0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        self._lagrangian_multiplier_return = self._lambda + self.rho * (self.Jc + self._xi)
        return self._lagrangian_multiplier_return

    def para_update(self, ep_cost_avg: float) -> None:
        self.Jc = ep_cost_avg
        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = min(0, self._cost_limit - ep_cost_avg)
        elif self.update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit > 1.1:
            self.sigma *= self._sigma_p
            self.sigma = min(self._max_sigma, self.sigma)
            self._min_xi = min(0, max(self._min_xi, self._xi))
        if self.update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit < 1.1:
            self.sigma *= self._sigma_d

        if float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho) > 0:
            new_xi = float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - ep_cost_avg + self._cost_limit - self._lambda/self._rho))

        self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi
        # self._xi = max(self._min_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

        delta_lambda = float(self._xi + ep_cost_avg - self._cost_limit)
        if delta_lambda < 0:
            self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr * 2)
        else:
            self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        self._lambda = min(self._lambda, self._penalty_max)

        # self._cost_ds.append(self._cost_d)

# # pylint: disable-next=too-few-public-methods,too-many-instance-attributes
# class ADMM2(abc.ABC):  # noqa: B024
#     """ADMM

#     Args:
#     """

#     # pylint: disable-next=too-many-arguments
#     def __init__(
#         self,
#         rho: float,
#         lambda_lr: float,
#         xi_lr: float,
#         penalty_max: int,
#         lagrangian_multiplier_init: float,
#         cost_limit: float,
#         sigma_init: float,
#         sigma_p: float,
#         sigma_d: float,
#         rho_steps: int,
#         max_sigma: float,
#         # lambda_optimizer: str,
#         d_delay: int,
#         pid_delta_p_ema_alpha: float,
#         pid_delta_d_ema_alpha: float,
#         sum_norm: bool,
#         diff_norm: bool,
#     ) -> None:
#         """Initialize an instance of :class:`ADMM`."""
#         lambda_optimizer = "Adam"
#         self._rho: float = rho
#         self._xi_rate: float = xi_lr
#         self._lambda_lr: float = lambda_lr
#         self.sigma: float = sigma_init
#         self._penalty_max: int = penalty_max
#         self._cost_limit: float = cost_limit
#         self._rho_steps: int = rho_steps
#         self._update_steps: int = 0
#         self._sigma_p: float = sigma_p
#         self._sigma_d: float = sigma_d
#         self._max_sigma: float = max_sigma
#         self._xi: float = 0.0
#         self.Jc: float = 0

#         # lambda
#         init_value = max(lagrangian_multiplier_init, 0.0)
#         self._lambda: torch.nn.Parameter = torch.nn.Parameter(
#             torch.as_tensor(init_value),
#             requires_grad=True,
#         )
#         self.lambda_range_projection: torch.nn.ReLU = torch.nn.ReLU()
#         assert hasattr(
#             torch.optim,
#             lambda_optimizer,
#         ), f'Optimizer={lambda_optimizer} not found in torch.'
#         torch_opt = getattr(torch.optim, lambda_optimizer)
#         self.lambda_optimizer: torch.optim.Optimizer = torch_opt(
#             [
#                 self._lambda,
#             ],
#             lr=lambda_lr,
#         )

#         # # xi
#         # self._xi: torch.nn.Parameter = torch.nn.Parameter(
#         #     torch.as_tensor(0.0),
#         #     requires_grad=True,
#         # )
#         # self.xi_penalty_function: torch.nn.ReLU = torch.nn.ReLU()
#         # assert hasattr(
#         #     torch.optim,
#         #     lambda_optimizer,
#         # ), f'Optimizer={lambda_optimizer} not found in torch.'
#         # torch_opt = getattr(torch.optim, lambda_optimizer)
#         # self.xi_optimizer: torch.optim.Optimizer = torch_opt(
#         #     [
#         #         self._xi,
#         #     ],
#         #     lr=xi_lr,
#         # )

#     @property
#     def lagrangian_multiplier(self) -> float:
#         """The lagrangian multiplier."""
#         return self._lambda

#     def init_xi(self, xi: float) -> float:
#         self._xi.data = torch.Tensor([xi])

#     def compute_lambda_loss(self, delta_lambda: float) -> torch.Tensor:
#         """Penalty loss for Lagrange multiplier.

#         .. note::
#             ``mean_ep_cost`` is obtained from ``self.logger.get_stats('EpCosts')[0]``, which is
#             already averaged across MPI processes.

#         Args:
#             mean_ep_cost (float): mean episode cost.

#         Returns:
#             Penalty loss for Lagrange multiplier.
#         """
#         return -self._lambda * delta_lambda

#     # def compute_xi_loss(self, ep_cost_avg: float) -> torch.Tensor:
#     #     return self.sigma * self.xi_penalty_function(-self._xi) + self._rho / 2 * (ep_cost_avg - self._cost_limit + self._xi + self._lambda/self._rho)**2

#     def para_update(self, ep_cost_avg: float) -> None:
#         self.Jc = ep_cost_avg
#         self._update_steps += 1
#         if self._update_steps == 1:
#             self._xi = min(0, self._cost_limit - ep_cost_avg)
#         if self._update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit >= 1.1:
#             self.sigma *= self._sigma_p
#             self.sigma = min(self._max_sigma, self.sigma)
#         elif self._update_steps % self._rho_steps == 0 and float(ep_cost_avg)/self._cost_limit < 1.1:
#             self.sigma *= self._sigma_d

#         if float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho) > 0:
#             new_xi = float(- ep_cost_avg + self._cost_limit - self._lambda/self._rho)
#         else:
#             new_xi = min(0, float(self.sigma/self._rho - ep_cost_avg + self._cost_limit - self._lambda/self._rho))
#         self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi

#         delta_lambda = float(self._xi + ep_cost_avg - self._cost_limit)
#         self.lambda_optimizer.zero_grad()
#         lambda_loss = self.compute_lambda_loss(delta_lambda)
#         lambda_loss.backward()
#         self.lambda_optimizer.step()
#         self._lambda.data.clamp_(
#             0.0,
#             self._penalty_max,
#         )  # enforce: lambda in [0, inf]


# pylint: disable-next=too-few-public-methods,too-many-instance-attributes
class ADMM3(abc.ABC):  # noqa: B024
    """ADMM

    Similar to the :class:`Lagrange` module, this module implements the PID version of the
    lagrangian method.

    Args:
        pid_kp (float): The proportional gain of the PID controller.
        pid_ki (float): The integral gain of the PID controller.
        pid_kd (float): The derivative gain of the PID controller.
        pid_d_delay (int): The delay of the derivative term.
        pid_delta_p_ema_alpha (float): The exponential moving average alpha of the delta_p.
        pid_delta_d_ema_alpha (float): The exponential moving average alpha of the delta_d.
        sum_norm (bool): Whether to use the sum norm.
        diff_norm (bool): Whether to use the diff norm.
        penalty_max (int): The maximum penalty.
        lagrangian_multiplier_init (float): The initial value of the lagrangian multiplier.
        cost_limit (float): The cost limit.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        d_delay: int,
        pid_delta_p_ema_alpha: float,
        pid_delta_d_ema_alpha: float,
        sum_norm: bool,
        diff_norm: bool,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
        sigma_p: float,
        sigma_d: float,
        rho_steps: int,
        max_sigma: float
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self.d_delay = d_delay
        self._pid_delta_p_ema_alpha: float = pid_delta_p_ema_alpha
        self._pid_delta_d_ema_alpha: float = pid_delta_d_ema_alpha
        self._penalty_max: int = penalty_max
        self._sum_norm: bool = sum_norm
        self._diff_norm: bool = diff_norm
        self._lambda: float = lagrangian_multiplier_init
        self._cost_ds: deque[float] = deque(maxlen=self.d_delay)
        self._cost_ds.append(0.0)
        self._delta_p: float = 0.0
        self._cost_d: float = 0.0
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self._rho_steps: int = rho_steps
        self.update_steps: int = 0
        self._sigma_p: float = sigma_p
        self._sigma_d: float = sigma_d
        self._max_sigma: float = max_sigma
        self._min_xi: float = -1000
        self.Lc: float = 0
        #--------- ADD more attributes-----
        self._lagrangian_multiplier_return = 0
        self._delta_sigma = (self._max_sigma - sigma_init)/(50 * 40)
        self._max_xi = 5

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        # self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        return self._lagrangian_multiplier_return

    def para_update(self, Lc: float) -> None:
        self.Lc = Lc
        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = -Lc

        # ---------------- v1 -------------------
        if self.update_steps % self._rho_steps == 0 and float(Lc) > 0:
            self.sigma *= self._sigma_p
            self.sigma = min(self._max_sigma, self.sigma)
        if self.update_steps % self._rho_steps == 0 and float(Lc) < 0:
            self.sigma *= self._sigma_d

        # ---------------- v2 -------------------
        # if self.update_steps % self._rho_steps == 0:
        #     self.sigma += self._delta_sigma
        #     self.sigma = min(self._max_sigma, self.sigma)


        if float(-Lc - self._lambda/self._rho) > 0:
            new_xi = float(- Lc - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))
        # ----------------------------- v1 ---------------------------------
        self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi

        # ----------------------------- v2 ---------------------------------
        # self._xi = min(self._max_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)
        # self._xi = max(self._min_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

        delta_lambda = float(self._xi + Lc)
        self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        # if delta_lambda < 0:
        #     self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        # else:
        #     self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        self._lambda = min(self._lambda, self._penalty_max)

        # self._cost_ds.append(self._cost_d)


class ADMM4(abc.ABC):
    """ADMM

    Args:

    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
        sigma_p: float,
        max_sigma: float,
        sigma_warmup: float,
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self._penalty_max: int = penalty_max
        self._lambda: float = lagrangian_multiplier_init
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self.update_steps: int = 0
        self._sigma_p: float = sigma_p
        self._max_sigma: float = max_sigma
        self._min_xi: float = -1000
        self._lagrangian_multiplier_return = 0
        self._delta_sigma = (self._max_sigma - sigma_init)/sigma_warmup
        self._max_xi = 5
        self.Lc =  0.0

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        # self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        return self._lagrangian_multiplier_return

    def para_update(self, Jc: float) -> None:
        Lc = Jc - self._cost_limit
        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = -Lc

        if float(-Lc - self._lambda/self._rho) > 0:
            new_xi = float(- Lc - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))
        # self._xi = new_xi
        # self._xi = min(self._max_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

        self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi

        delta_lambda = float(self._xi + Lc)
        self._lambda_resisual = delta_lambda * self._lambda_lr
        # self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        self._lambda = self._lambda + delta_lambda * self._lambda_lr
        # self._lambda = max(0.0, self._lambda + delta_lambda * self._rho)
        # self._lambda_resisual = delta_lambda * self._rho
        # self._lambda_resisual = delta_lambda * self._lambda_lr
        # self._lambda = self._lambda + delta_lambda * self._rho

        # self._lambda = min(self._lambda, self._penalty_max)
        # self._lambda = self._lambda, self._penalty_max

    def update_sigma(self) -> None:
         self.sigma = 5
        # self.sigma += self._delta_sigma
        # self.sigma = min(self._max_sigma, self.sigma)




# class ADMM4(abc.ABC):
#     """ADMM

#     Args:

#     """

#     # pylint: disable-next=too-many-arguments
#     def __init__(
#         self,
#         rho: float,
#         lambda_lr: float,
#         xi_lr: float,
#         lagrangian_multiplier_init: float,
#         cost_limit: float,
#         sigma: float,
#     ) -> None:
#         """Initialize an instance of :class:`ADMM`."""
#         self._rho: float = rho
#         self._xi_rate: float = xi_lr
#         self._lambda_lr: float = lambda_lr
#         self.sigma: float = sigma
#         self._lambda: float = lagrangian_multiplier_init
#         self._cost_limit: float = cost_limit
#         self._xi: float = 0.0
#         self.update_steps: int = 0
#         self._min_xi: float = -1000
#         self._lagrangian_multiplier_return = 0
#         self._max_xi = 5

#     @property
#     def lagrangian_multiplier(self) -> float:
#         """The lagrangian multiplier."""
#         self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
#         return self._lagrangian_multiplier_return

#     def para_update(self, Lc: float) -> None:
#         # update z, y or y z y instead of update y ,z

#         self.update_steps += 1
#         if self.update_steps == 1:
#             self._xi = -Lc
#         # -------------------------------------------------
#         if float(-Lc - self._lambda/self._rho) > 0:
#             new_xi = float(- Lc - self._lambda/self._rho)
#         else:
#             new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))
#         new_xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi

#         self._xi = min(self._max_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

#         delta_lambda = float(self._xi + Lc)
#         self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)




#         # self._xi_rate = self._xi_rate / 2
#         # self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi
#         # -------------------------------------------------

#         # delta_lambda = float(new_xi + Lc)
#         # self._lambda = self._lambda + delta_lambda * self._lambda_lr

#         # -------------------------------------------------
#         # if self._rho < 0.01:
#         #     self._rho = 2 * self._rho
#         # elif self._rho >= 0.02:
#         #     self._rho = self._rho / 2
#         # else:
#         #     self._rho = self._rho
#         # -------------------------------------------------------
#         # self._xi = new_xi
#         # -------------------------------------------------------
# #         if float(-Lc - self._lambda/self._rho) > 0:
# #             new_xi = float(- Lc - self._lambda/self._rho)
# #         else:
# #             new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))

# #         self._xi = (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi




#     def update_sigma(self) -> None:
#         self.sigma = self.sigma
#         # self.sigma += self._delta_sigma
#         # self.sigma = min(self._max_sigma, self.sigma)




class ADMM5(abc.ABC):
    """ADMM

    Args:

    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        rho: float,
        lambda_lr: float,
        xi_lr: float,
        penalty_max: int,
        lagrangian_multiplier_init: float,
        cost_limit: float,
        sigma_init: float,
        sigma_p: float,
        max_sigma: float,
        sigma_warmup: float,
    ) -> None:
        """Initialize an instance of :class:`ADMM`."""
        self._rho: float = rho
        self._xi_rate: float = xi_lr
        self._lambda_lr: float = lambda_lr
        self.sigma: float = sigma_init
        self._penalty_max: int = penalty_max
        self._lambda: float = lagrangian_multiplier_init
        self._cost_limit: float = cost_limit
        self._xi: float = 0.0
        self.update_steps: int = 0
        self._sigma_p: float = sigma_p
        self._max_sigma: float = max_sigma
        self._min_xi: float = -1000
        self._lagrangian_multiplier_return = 0
        self._delta_sigma = (self._max_sigma - sigma_init)/sigma_warmup
        self._max_xi = 5

    @property
    def lagrangian_multiplier(self) -> float:
        """The lagrangian multiplier."""
        # self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        self._lagrangian_multiplier_return = self._lambda + self._rho * (self.Lc + self._xi)
        return self._lagrangian_multiplier_return

    def para_update(self, Lc: float) -> None:


        self.update_steps += 1
        if self.update_steps == 1:
            self._xi = -Lc

        if float(-Lc - self._lambda/self._rho) > 0:
            new_xi = float(- Lc - self._lambda/self._rho)
        else:
            new_xi = min(0, float(self.sigma/self._rho - Lc - self._lambda/self._rho))
        self._xi = min(self._max_xi, (1 - self._xi_rate) * self._xi + self._xi_rate * new_xi)

        delta_lambda = float(self._xi + Lc)
        self._lambda = max(0.0, self._lambda + delta_lambda * self._lambda_lr)
        self._lambda = min(self._lambda, self._penalty_max)

    def update_sigma(self) -> None:
        self.sigma += self._delta_sigma
        self.sigma = min(self._max_sigma, self.sigma)