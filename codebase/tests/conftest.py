import os

# Skip legacy PyTorch Longformer tests under tests/s5/*
# These depend on an external 'longformer' package and CUDA-only code that is not part of this JAX/BigBird codebase.
# Prevent collection to avoid import errors and irrelevant failures.

from pathlib import Path

# Skip legacy PyTorch Longformer tests under tests/s5/*
# These depend on an external 'longformer' package and CUDA-only code that is not part of this JAX/BigBird codebase.
# Prevent collection to avoid import errors and irrelevant failures.

def pytest_ignore_collect(collection_path: Path):
    try:
        path_str = str(collection_path)
    except Exception:
        path_str = str(collection_path)
    s5_segment = str(Path('tests') / 's5') + os.path.sep
    return s5_segment in path_str