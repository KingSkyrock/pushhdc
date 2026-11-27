"""PushHDC: evolving HDC encoders with PushGP.

`evolve` searches for Push programs that encode samples as hvs, and an
`Evaluator` scores each program by how well a simple classifier does on its
hvs. `choose` then picks one of the finalists of the run on new evaluators,
and `tune_ints` tries other values for the ints of the program that `choose`
picked. `program_text` prints a program.

The library has two halves that share only `fitness` and `kernel`. The search
works with any evaluator and never touches hvs or Push instructions:

- `evolution` holds `evolve`
- `choice` holds `choose` and `tune_ints`, which pick the program that a run
  gives back
- `dedup` scores each different encoding once, and `clearing` keeps programs
  that do something new ahead of copies of better programs
- `parallel` scores many genomes at a time in worker processes

The scoring turns a program into hvs and scores them:

- `evaluation` scores a program as an encoder, by cross-validated accuracy
  and cost
- `encoder_ops` gives the ops that an encoder can use and their costs, and
  makes random genes from them
- `recording` runs a program one time and replays its hv ops on each length
- `inputs` holds the inputs of an evaluator and cuts them by sample length,
  and `levels` gives each number a level and each level an hv, so that inputs
  can hold numbers
- `instructions`, `units` and `push_hdc` make the hv ops into Push
  instructions
- `hypervector` holds the hv type and its ops

`fitness` holds the score and what the search can ask of an evaluator, and
`kernel` tells how alike a program makes each pair of samples. `lint-imports`
checks that the imports keep to these layers (see `pyproject.toml`).
"""

from pushhdc.choice import choose, tune_ints
from pushhdc.encoder_ops import program_text
from pushhdc.evaluation import Evaluator, centroid_accuracy
from pushhdc.evolution import Result, evolve
from pushhdc.fitness import Fitness, ScoredGenome
from pushhdc.hypervector import HyperVector
from pushhdc.inputs import Lookup

__all__ = [
    "Evaluator",
    "Fitness",
    "HyperVector",
    "Lookup",
    "Result",
    "ScoredGenome",
    "centroid_accuracy",
    "choose",
    "evolve",
    "program_text",
    "tune_ints",
]
