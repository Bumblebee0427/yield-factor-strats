"""Factor model implementations."""

from ycurve.factors.ae import LinearAutoencoder
from ycurve.factors.mfa import fit_mfa
from ycurve.factors.pca import fit_pca

__all__ = ["fit_pca", "fit_mfa", "LinearAutoencoder"]
