# The "better than a textbook" standard

Every topic directory must contain:

1. `README.md` – the lesson. Order: motivation and the one-sentence idea -> notation table ->
   derivation with *every* algebra step shown (no "it can be shown") -> geometric/intuitive picture ->
   a tiny hand-computable worked example (numbers checked by the code) -> algorithm in pseudocode ->
   complexity -> failure modes and pitfalls -> 6+ exercises with full solutions.
   Math in LaTeX ($...$ / $$...$$). Link each equation to the code line that implements it.
2. `src/<topic>.py` – NumPy-only implementation, readable over clever, docstrings cite README equation numbers.
3. `tests/` – pytest. Verify against closed forms, numerical gradient checks, or sklearn (optional import, skipped if absent).
   Include property tests (invariants), not just examples.
4. `make_figures.py` – regenerates all figures in `figures/` deterministically (fixed seeds). Figures must teach
   (annotated, labelled axes), not decorate.
5. `demo.py` – runs end to end in <30 s and prints results.

Rules: Python 3, numpy + matplotlib only for the library/figures. Every numeric claim in the README must be
reproduced by code in the repo. Run the tests and the figure script before committing. Commit often on your branch.
