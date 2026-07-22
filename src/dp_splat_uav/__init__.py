"""DP-Splat-UAV: tiled Dirichlet-process Gaussian splatting for aerial mapping.

Application companion to DP-Splat; imports ``dp_splat`` and never modifies it.

float64 is mandatory: the natural-parameter and prior-subtraction algebra of the
tiled pipeline is catastrophically cancellative in float32, and the Beta/digamma
path loses ELBO monotonicity below double precision.
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.0.1"
