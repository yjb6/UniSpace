"""Config loading helpers.

Registers OmegaConf's built-in ``oc.env`` resolver so that config files can
reference environment variables with an optional default::

    encoder_config_path: ${oc.env:MODEL_ROOT,/path/to/models}/siglip2-base-patch16-256/main

``oc.env:VAR,default`` expands to ``os.environ['VAR']`` if set, otherwise to
``default`` (which may be empty). Importing this module (once, at program start)
makes the resolver available to every subsequent ``OmegaConf.load`` call.
"""

import os

from omegaconf import OmegaConf

# OmegaConf >= 2.2 ships ``oc.env`` as a built-in resolver, but it must be
# explicitly registered to be active. Registering twice is a no-op.
if not OmegaConf.has_resolver("oc.env"):
    OmegaConf.register_new_resolver("oc.env", lambda var, default=None: os.environ.get(var, default))


def load_config(path: str):
    """Load a YAML config with ``oc.env`` resolution enabled."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True) if False else cfg  # keep structured; resolve on demand
