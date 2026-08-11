import torch as th
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde
import logging
logger = logging.getLogger(__name__)

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()
    SEMANTIC_RECON = enum.auto()  # time-dependent endpoint: x1(t) = s + (1-t)*z

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


def truncated_logitnormal_sample(
    shape, mu, sigma, low=0.0, high=1.0
):
    """
    Samples X in (0,1) with Z = logit(X) ~ Normal(mu, sigma^2), truncated so X in [low, high].
    Works for scalars or tensors mu/sigma/low/high with broadcasting.

    Args:
        shape: output batch shape (e.g., (N,) or (N,M)). Leave () to broadcast to mu.shape.
        mu, sigma: tensors or floats (sigma > 0).
        low, high: truncation bounds in [0,1]. (low can be 0, high can be 1).
        device, dtype: optional overrides.

    Returns:
        Tensor of samples with shape = broadcast(shape, mu.shape, ...)
    """
    mu   = th.as_tensor(mu)
    sigma= th.as_tensor(sigma)
    low  = th.as_tensor(low)
    high = th.as_tensor(high)

    # Map truncation bounds to logit space; handles 0/1 → ±inf automatically.
    z_low  = th.logit(low)   # = -inf if low==0
    z_high = th.logit(high)  # = +inf if high==1

    # Standardize bounds for the base Normal(0,1)
    base = th.distributions.Normal(th.zeros_like(mu), th.ones_like(sigma))
    alpha = (z_low  - mu) / sigma
    beta  = (z_high - mu) / sigma

    # Truncated-normal inverse CDF sampling:
    # U ~ Uniform(Φ(alpha), Φ(beta));  Z = mu + sigma * Φ^{-1}(U);  X = sigmoid(Z)
    cdf_alpha = base.cdf(alpha)
    cdf_beta  = base.cdf(beta)

    # Draw uniforms on the truncated interval
    out_shape = th.broadcast_shapes(shape, mu.shape, sigma.shape, low.shape, high.shape)
    U = th.rand(out_shape, device=mu.device, dtype=mu.dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp_(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = th.sigmoid(Z)

    # Numerical safety when low/high are extremely close; clamp back into [low, high].
    return X.clamp(low, high)


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        time_dist_type,
        time_dist_shift,
        train_eps,
        sample_eps,
        use_time_shift,
        sem_dim=None,
        recon_loss_weight=1.0,
        normalize_channel_weight=False,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
            PathType.SEMANTIC_RECON: path.SemanticReconPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        assert self.time_dist_shift >= 1.0, "time distribution shift must be >= 1.0."
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps
        self.use_time_shift = use_time_shift
        self.sem_dim = sem_dim
        self.recon_loss_weight = float(recon_loss_weight)
        self.normalize_channel_weight = normalize_channel_weight
        if self.sem_dim is not None:
            logger.info(f"Channel-wise loss: sem_dim={self.sem_dim}, recon_loss_weight={self.recon_loss_weight}, normalize={self.normalize_channel_weight}")

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)


    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1 - 1 / 1000
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan, path.SemanticReconPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """

        x0 = th.randn_like(x1)
        dist_options = self.time_dist_type.split("_")
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        if dist_options[0] == "uniform":
            t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
            # print('UNIFORM IS CALLED')
        elif dist_options[0] == "logit-normal":
            assert len(dist_options) == 3, "Logit-normal distribution must specify the mean and variance."
            mu, sigma = float(dist_options[1]), float(dist_options[2])
            assert sigma > 0, "Logit-normal distribution must have positive variance."
            t = truncated_logitnormal_sample(
                (x1.shape[0],), mu=mu, sigma=sigma, low=t0, high=t1
            )
            # print('LOGITNORMAL IS CALLED')
        elif dist_options[0] == "fixed":
            # Fixed t for 1-step training (e.g., "fixed_1.0")
            fixed_t = float(dist_options[1]) if len(dist_options) > 1 else 1.0
            t = th.full((x1.shape[0],), fixed_t)
        else:
            raise NotImplementedError(f"Unknown time distribution type {self.time_dist_type}")

        t = t.to(x1)
        # logger.debug(f"t before shift: {t}")
        #sqrt_size_ratio = 1 / self.time_dist_shift # already sqrted
        if self.use_time_shift:
            t = self.time_dist_shift * t / (1 + (self.time_dist_shift - 1) * t)
        # logger.debug(f"t after shift: {t}")
        return t, x0, x1


    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}

        # Extract auxiliary loss parameters so they aren't passed to the model's forward pass
        aux_losses_cfg = model_kwargs.pop('aux_losses_cfg', {})
        latent_var = model_kwargs.pop('latent_var', None)

        # Pre-emptively pop all potential projection modules to avoid passing them to the model
        proj_modules = {}
        keys_to_pop = [k for k in model_kwargs.keys() if k.endswith('_proj')]
        for k in keys_to_pop:
            proj_name = k.replace('_proj', '')
            proj_modules[proj_name] = model_kwargs.pop(k)

        # Handle legacy 'semetic_proj' key if present
        if 'semetic_proj' in model_kwargs:
            proj_modules['semantic'] = model_kwargs.pop('semetic_proj')

        # Pop semantic/recon components for SemanticReconPlan path construction.
        # These must NOT be forwarded to the model's forward().
        path_semantic = model_kwargs.pop('path_semantic', None)
        path_recon    = model_kwargs.pop('path_recon', None)

        logger.debug(f"x1 shape{x1.shape}")
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1, semantic=path_semantic, recon=path_recon)
        logger.debug(f"xt shape{xt.shape}")
        model_output = model(xt, t, **model_kwargs)
        logger.debug(f"model output shape{model_output.shape}")
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        terms['xt'] = xt
        terms['t'] = t
        terms['x0'] = x0
        if self.model_type == ModelType.VELOCITY:
            mse = (model_output - ut) ** 2
            if self.sem_dim is not None and self.sem_dim > 0:
                # Channel dim is dim=1 for [B,C,H,W] or dim=-1 for [B,N,C]
                if mse.dim() == 4:  # [B, C, H, W]
                    terms['loss_sem'] = mean_flat(mse[:, :self.sem_dim])
                    terms['loss_recon'] = mean_flat(mse[:, self.sem_dim:])
                    if self.recon_loss_weight != 1.0:
                        C = mse.shape[1]
                        recon_dim = C - self.sem_dim
                        sem_w = 1.0
                        recon_w = self.recon_loss_weight
                        if self.normalize_channel_weight:
                            norm_factor = C / (self.sem_dim * sem_w + recon_dim * recon_w)
                            sem_w *= norm_factor
                            recon_w *= norm_factor
                        weight = th.ones(1, C, 1, 1, device=mse.device)
                        weight[:, :self.sem_dim] = sem_w
                        weight[:, self.sem_dim:] = recon_w
                        mse = mse * weight
                else:  # [B, N, C]
                    terms['loss_sem'] = mean_flat(mse[..., :self.sem_dim])
                    terms['loss_recon'] = mean_flat(mse[..., self.sem_dim:])
                    if self.recon_loss_weight != 1.0:
                        C = mse.shape[-1]
                        recon_dim = C - self.sem_dim
                        sem_w = 1.0
                        recon_w = self.recon_loss_weight
                        if self.normalize_channel_weight:
                            norm_factor = C / (self.sem_dim * sem_w + recon_dim * recon_w)
                            sem_w *= norm_factor
                            recon_w *= norm_factor
                        weight = th.ones(1, 1, C, device=mse.device)
                        weight[..., :self.sem_dim] = sem_w
                        weight[..., self.sem_dim:] = recon_w
                        mse = mse * weight
            terms['loss'] = mean_flat(mse)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()

            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        # --- Compute Auxiliary Losses (e.g., Semantic Loss) ---
        if self.model_type == ModelType.VELOCITY and aux_losses_cfg:
            # Apply base weight to the original loss
            base_weight = aux_losses_cfg.get('base_weight', 1.0)
            logger.debug(f"flow loss: {terms['loss'].detach().mean().item()}")
            terms['loss'] = terms['loss'] * base_weight

            # Prepare unnormalized velocity if any projection loss is enabled
            # We check if any proj loss is enabled to avoid unnecessary computation
            proj_losses_cfg = aux_losses_cfg.get('proj_losses', {})
            need_unnorm = any(
                isinstance(cfg, dict) and cfg.get('enabled', False)
                for cfg in proj_losses_cfg.values()
            )

            v_pred_unnorm, v_gt_unnorm = None, None
            if need_unnorm and latent_var is not None:
                logger.debug(f"model_output shape: {model_output.shape}")
                logger.debug(f"latent_var shape: {latent_var.shape}")

                # 1. Convert model_output and ut to [B, N, C] if they are [B, C, H, W]
                if model_output.dim() == 4:
                    B, C, H, W = model_output.shape
                    N = H * W
                    # [B, C, H, W] -> [B, C, N] -> [B, N, C]
                    v_pred_flat = model_output.view(B, C, N).transpose(1, 2)
                    v_gt_flat = ut.view(B, C, N).transpose(1, 2)
                else:
                    v_pred_flat = model_output
                    v_gt_flat = ut
                    B, N, C = v_pred_flat.shape

                # 2. Reshape latent_var to [1, N, C] or [1, 1, C]
                if latent_var.dim() == 1:
                    # [C] -> [1, 1, C]
                    scale = th.sqrt(latent_var + 1e-5).view(1, 1, -1)
                elif latent_var.dim() == 2:
                    # [N, C] -> [1, N, C]
                    scale = th.sqrt(latent_var + 1e-5).unsqueeze(0)
                elif latent_var.dim() == 3:
                    # [C, H, W] -> [C, N] -> [N, C] -> [1, N, C]
                    scale = th.sqrt(latent_var + 1e-5).view(latent_var.shape[0], -1).transpose(0, 1).unsqueeze(0)
                else:
                    scale = th.sqrt(latent_var + 1e-5)

                # 3. Unnormalize
                v_pred_unnorm = v_pred_flat * scale
                v_gt_unnorm = v_gt_flat * scale

            # --- Loop over projection losses ---
            # This makes it easy to add more projection losses in the future
            for proj_name, proj_cfg in proj_losses_cfg.items():
                if isinstance(proj_cfg, dict) and proj_cfg.get('enabled', False):
                    # Get the projection module from our pre-popped dict
                    proj_module = proj_modules.get(proj_name)

                    if proj_module is not None:
                        if v_pred_unnorm is None:
                            logger.warning(f"{proj_name} loss enabled but latent_var is missing.")
                            continue

                        # Map to projection space
                        proj_v_pred = proj_module(v_pred_unnorm)
                        proj_v_gt = proj_module(v_gt_unnorm)

                        # Compute MSE
                        proj_loss = mean_flat(((proj_v_pred - proj_v_gt) ** 2))

                        logger.debug(f"{proj_name}_loss: {proj_loss.detach().mean().item()}")
                        weight = proj_cfg.get('weight', 1.0)

                        # Accumulate to total loss
                        terms['loss'] = terms['loss'] + weight * proj_loss
                        terms[f'{proj_name}_loss'] = proj_loss.detach()
                    else:
                        logger.warning(f"{proj_name} loss enabled but {proj_name}_proj not found in model_kwargs")
        # logger.debug(terms)
        return terms


    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn


    def get_score(
        self,
    ):
        """member function for obtaining score of
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def sde_diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        def sde_drift_fn(x, t, model, **kwargs):
            drift_mean = self.drift(x, t, model, **kwargs) - sde_diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
            return drift_mean


        return sde_drift_fn, sde_diffusion_fn

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x - self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
            time_dist_shift=self.transport.time_dist_shift,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)


        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * (1 - t1)
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        **kwargs
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        # print(t0, t1)
        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            time_dist_shift=self.transport.time_dist_shift,
            use_time_shift=kwargs.get("use_time_shift", True),
        )

        return _ode.sample