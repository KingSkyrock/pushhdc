from pushhdc.pyshgp.push.atoms import CodeBlock, Literal
from pushhdc.pyshgp.push.interpreter import PushInterpreter
from pushhdc.pyshgp.push.program import Program, ProgramSignature
from pushhdc.pyshgp.push.type_library import PushTypeLibrary


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
