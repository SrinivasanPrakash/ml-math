# ml-math

Five core machine-learning algorithms, derived from first principles and built with NumPy only.
Each lesson follows the same standard (see [STANDARD.md](STANDARD.md)): a full derivation, a hand-checkable
worked example, a from-scratch implementation, tests, figures and solved exercises.
Every number quoted in a lesson is reproduced by code in the repo.

| # | Lesson | What it derives | Tests | Figures |
|---|--------|-----------------|-------|---------|
| 1 | [Linear regression from zero](01_linear_regression/README.md) | Normal equations (calculus and projection views), gradient descent and conditioning, QR/SVD solvers, ridge as shrinkage and MAP | 32 | 8 |
| 2 | [Principal component analysis](02_pca/README.md) | Variance maximisation = minimum reconstruction error, SVD link, three solvers, probabilistic PCA | 22 | 6 |
| 3 | [Neural network from scratch](03_neural_network/README.md) | Backprop in batch matrix form, softmax + cross-entropy gradient, initialisation, SGD/Momentum/Adam | 35 | 6 |
| 4 | [Markov chain text generator](04_markov_text/README.md) | MLE of transition probabilities, stationary distribution, smoothing, perplexity vs order | 35 | 5 |
| 5 | [Decision tree](05_decision_tree/README.md) | Impurity and information gain, O(n log n) split search, cost-complexity pruning, bagging | 71 | 9 |

## Running a lesson

Each directory is self-contained. From inside it:

```bash
python3 -m pytest -q tests     # tests
python3 demo.py                # prints the numbers quoted in the README
python3 make_figures.py        # regenerates figures/
```

If the demos feel slow, pin BLAS to one thread first:
`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Requires Python 3, NumPy, matplotlib and pytest; scikit-learn is optional (cross-checks are skipped without it).
