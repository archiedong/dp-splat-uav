import sys, pathlib
# dp-splat (Paper 1, frozen at arXiv v1) is imported by path until the pinned release lands
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dp-splat" / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import jax

# float64 everywhere, before any array is created (see dp_splat_uav.__init__)
jax.config.update("jax_enable_x64", True)
