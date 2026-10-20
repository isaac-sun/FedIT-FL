import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data", "databricks-dolly-15k.jsonl")


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from tests.tiny_model import build_tiny_model
    return build_tiny_model(str(tmp_path_factory.mktemp("tiny_qwen2")), DATA)
