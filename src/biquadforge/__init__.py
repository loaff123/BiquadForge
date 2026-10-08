"""Finite-suite standard scalar CMSIS Q15 qualification."""
from .spec import BiquadForgeError, BuildSpec, load_spec

__version__ = "0.1.0"
__all__ = ["BiquadForgeError", "BuildSpec", "load_spec"]
from .fixedpoint import Q15Cascade
__all__.append("Q15Cascade")
from .qualification import BuildResult, qualify
__all__ += ["BuildResult", "qualify"]
from .pack import write_pack
__all__.append("write_pack")
from .verify import VerificationResult, verify_pack
__all__ += ["VerificationResult", "verify_pack"]
