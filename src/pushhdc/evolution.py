import collections
import collections.abc

# PyshGP 0.1.8 imports Sequence from collections, where it was removed in Python 3.10.
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence  # type: ignore[attr-defined]

from pyshgp.push.atoms import CodeBlock, Literal  # type: ignore[import-untyped]
from pyshgp.push.interpreter import PushInterpreter  # type: ignore[import-untyped]
from pyshgp.push.program import Program, ProgramSignature  # type: ignore[import-untyped]
from pyshgp.push.type_library import PushTypeLibrary  # type: ignore[import-untyped]


def run_pyshgp_smoke_test() -> list[int]:
    """Run a minimal Push program and return its integer output."""
    type_library = PushTypeLibrary()
    program = Program(
        code=CodeBlock([Literal(value=42, push_type=type_library["int"])]),
        signature=ProgramSignature(arity=0, output_stacks=["int"]),
    )
    output = PushInterpreter().run(program, inputs=[])
    if output != [42]:
        raise RuntimeError(f"Unexpected PyshGP output: {output!r}")
    return [int(value) for value in output]


if __name__ == "__main__":
    run_pyshgp_smoke_test()
