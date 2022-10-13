from semissl.semi.base import base_semi_wrapper
from semissl.semi.correlative import correlative_semi_wrapper
from semissl.semi.cross_entropy import cross_entropy_semi_wrapper

__all__ = [
    "base_semi_wrapper",
    "correlative_semi_wrapper",
    "cross_entropy_semi_wrapper",
]

SEMISUPERVISED = {
    "base": base_semi_wrapper,
    "correlative": correlative_semi_wrapper,
    "cross_entropy": cross_entropy_semi_wrapper,
}
