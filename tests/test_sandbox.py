"""The code sandbox. The Docker tests run only where the sandbox image is built."""
import pytest

from drugagent.sandbox import docker_available, run_python

needs_docker = pytest.mark.skipif(not docker_available(), reason="docker sandbox image not built")


def test_outputs_written_to_out_come_back():
    result = run_python("import json\nopen('out/r.json','w').write(json.dumps({'n': 42}))\nprint('done')",
                        backend="subprocess")
    assert result.ok and result.stdout.strip() == "done"
    assert result.outputs == {"r.json": b'{"n": 42}'}
    assert not result.isolated  # the subprocess backend never claims to be a boundary


def test_a_runaway_loop_is_stopped():
    result = run_python("while True:\n    pass", timeout=2, backend="subprocess")
    assert result.timed_out and not result.ok


def test_input_files_are_provided():
    result = run_python("print(open('data.csv').read().count(','))", files={"data.csv": b"a,b,c\n1,2,3\n"},
                        backend="subprocess")
    assert result.stdout.strip() == "4"


@pytest.mark.docker
@needs_docker
def test_docker_blocks_the_network_and_the_filesystem():
    code = """
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
    print("network: open")
except OSError:
    print("network: blocked")
try:
    open("/etc/evil", "w")
    print("root fs: writable")
except OSError:
    print("root fs: read-only")
import os
print("uid:", os.getuid())
"""
    result = run_python(code, backend="docker", timeout=60)
    assert result.isolated and result.ok, result.stderr
    assert "network: blocked" in result.stdout
    assert "root fs: read-only" in result.stdout
    assert "uid: 65534" in result.stdout


@pytest.mark.docker
@needs_docker
def test_docker_timeout_kills_the_container():
    result = run_python("while True:\n    pass", backend="docker", timeout=8)
    assert result.timed_out


@pytest.mark.docker
@needs_docker
def test_docker_has_the_scientific_stack_and_returns_a_figure():
    code = """
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
plt.plot([1, 2], [3, 4]); plt.savefig("out/fig.png")
print(round(Descriptors.MolWt(Chem.MolFromSmiles("CCO")), 2))
"""
    result = run_python(code, backend="docker", timeout=120)
    assert result.ok, result.stderr
    assert result.stdout.strip() == "46.07" and "fig.png" in result.outputs
