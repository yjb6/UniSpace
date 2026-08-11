import torch
from torch import nn

import logging


logger = logging.getLogger(__name__)


def make_xpred_sampler(num_steps: int, sampling_method: str = "euler", t_eps: float = 1e-3):
    """
    Create a JiT-style x-pred sampler operating in latent space.

    The underlying model is assumed to output x_pred given (z_t, t, **kwargs),
    and we convert it to a velocity field via:

        v_pred = (x_pred - z_t) / (1 - t)

    The returned object has the same callable interface as the existing
    Transport Sampler ODE/SDE samplers:

        xs = sampler(z_init, model_fn, **model_kwargs)
        final_z = xs[-1]
    """
    method = sampling_method.lower()
    if method not in ("euler", "heun"):
        raise NotImplementedError(f"x-pred sampler only supports euler/heun, got {sampling_method}")

    def _forward_velocity(z: torch.Tensor, t: torch.Tensor, model_fn, model_kwargs):
        """
        Map x-pred model output to velocity:

            v_pred = (x_pred - z) / (1 - t)
        """
        bsz = z.size(0)
        if t.ndim == 0:
            t = t.expand(bsz)
        t = t.to(z.device)
        t_view = t.view(bsz, *([1] * (z.ndim - 1)))
        x_pred = model_fn(z, t, **model_kwargs)
        v_pred = (x_pred - z) / (1.0 - t_view).clamp_min(t_eps)
        return v_pred

    def _sample(init: torch.Tensor, model_fn, **model_kwargs):
        """
        Fixed-step ODE sampler (Euler / Heun) for JiT-style x-pred velocity.

        Args:
            init: initial latent z_0
            model_fn: callable taking (z_t, t, **model_kwargs) and returning x_pred
            model_kwargs: passed through to model_fn
        """
        z = init
        xs = []
        bsz = z.size(0)
        device = z.device
        ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            if method == "euler":
                v_pred = _forward_velocity(z, t, model_fn, model_kwargs)
                z = z + dt * v_pred
            else:  # heun
                v_t = _forward_velocity(z, t, model_fn, model_kwargs)
                z_euler = z + dt * v_t
                t_next = ts[i + 1].expand(bsz)
                v_next = _forward_velocity(z_euler, t_next, model_fn, model_kwargs)
                v_pred = 0.5 * (v_t + v_next)
                z = z + dt * v_pred

            xs.append(z)

        return xs

    return _sample


def compute_xpred_loss(
    model: nn.Module,
    x: torch.Tensor,
    labels: torch.Tensor,
    *,
    transport=None,
    t_eps: float = 1e-3,
    sample_t_method: str = "transport",
) -> torch.Tensor:
    """
    JiT-style x-pred loss in latent space.

    Given latent "clean" target x, we construct:

        z_t = t * x + (1 - t) * e
        v    = (x      - z_t) / (1 - t)
        x̂    = model(z_t, t, y=labels)
        v̂    = (x_hat - z_t) / (1 - t)
        L    = E[ ||v - v̂||^2 ]
    """
    device = x.device
    bsz = x.size(0)

    # If a Transport object is provided, reuse its time sampler so that
    # x-pred shares the same time distribution (uniform/logit-normal,
    # eps truncation, time_dist_shift, etc.) as the underlying FM setup.
    # Linear plan is parameterized data->noise; JiT is noise->data,
    # so we flip t -> 1 - t to keep the *geometric* end with higher
    # sampling density consistent.
    if sample_t_method == "transport" and transport is not None:
        # transport.sample(x1) -> (t, x0, x1)
        t, _, _ = transport.sample(x)
        t = 1.0 - t
    elif sample_t_method == "normal_sigmoid":
        t = torch.randn(bsz, device=device)*0.8 - 0.8
        t = torch.sigmoid(t)
    else:
        raise NotImplementedError(f"Invalid sample_t_method: {sample_t_method}")
    logger.debug(f"t: {t}")
    t_view = t.view(bsz, *([1] * (x.ndim - 1)))
    e = torch.randn_like(x)
    z_t = t_view * x + (1.0 - t_view) * e

    v = (x - z_t) / (1.0 - t_view).clamp_min(t_eps)
    x_pred = model(z_t, t, y=labels)
    v_pred = (x_pred - z_t) / (1.0 - t_view).clamp_min(t_eps)

    loss = (v - v_pred) ** 2
    logger.debug(f"xpred loss: {loss}")
    return loss.mean()

