import socket

import pytest


@pytest.fixture(scope="session")
def showdown():
    """Skip tests that play real battles when the local Showdown server is not running."""
    try:
        socket.create_connection(("localhost", 8000), timeout=1).close()
    except OSError:
        pytest.skip("Showdown server not running on localhost:8000")
