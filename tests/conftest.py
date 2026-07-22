import jax

# Full-precision linear algebra is required throughout (same policy as dp-splat).
jax.config.update("jax_enable_x64", True)
