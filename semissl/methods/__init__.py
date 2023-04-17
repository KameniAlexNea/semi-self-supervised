from semissl.methods.barlow_twins import BarlowTwins
from semissl.methods.base import BaseModel
from semissl.methods.linear import LinearRecognitionModel as LinearModel
from semissl.methods.vicreg import VICReg

METHODS = {
    # base classes
    "base": BaseModel,
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "vicreg": VICReg,
}
__all__ = ["BarlowTwins", "BaseModel", "LinearModel", "VICReg"]

try:
    from semissl.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
