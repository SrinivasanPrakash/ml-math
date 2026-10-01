import os

# Small matrices: single-threaded BLAS is much faster than thread contention.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
