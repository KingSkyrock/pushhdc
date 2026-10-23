from pushhdc import hello
from pushhdc.evolution import run_pyshgp_smoke_test


def test_hello() -> None:
    assert hello() == "Hello from pushhdc!"


def test_pyshgp_smoke_test() -> None:
    assert run_pyshgp_smoke_test() == [42]
