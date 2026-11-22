"""PushHDC: evolving HDC encoders with PushGP.

`evolve` searches for Push programs that encode samples as hvs, and `choose`
picks one of the finalists that `evolve` gives back. The modules, from the
hv up to the search:

- `hypervector` holds the hv type and its ops
- `instructions`, `units` and `push_hdc` make the hv ops into Push
  instructions
- `encoder_ops` gives the ops that an encoder can use and their costs, and
  makes random genes from them
- `inputs` holds the inputs of an evaluator and cuts them by sample length
- `recording` runs a program one time and replays its hv ops on each length
- `evaluation` scores a program as an encoder, by cross-validated accuracy
  and cost
- `fitness` holds the score that scoring and the search share
- `dedup` scores each different encoding once, and `clearing` keeps programs
  that do something new ahead of copies of better programs
- `parallel` scores many genomes at a time in worker processes
- `evolution` holds `evolve` and `choose`
"""

from pushhdc.hypervector import HyperVector
from pushhdc.push_hdc import (
    HyperVectorPushType,
    hdc_instruction_set,
    hdc_type_library,
)

__all__ = [
    "HyperVector",
    "HyperVectorPushType",
    "hdc_instruction_set",
    "hdc_type_library",
]
