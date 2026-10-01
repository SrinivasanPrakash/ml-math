"""End-to-end demo; every number quoted in README.md is printed here. Runs in a few seconds."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")  # small matrices: threads only add overhead
import sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from pca import *

t0 = time.time()
np.set_printoptions(precision=4, suppress=True)

print("== 1. Worked example (README section 8) ==")
X = np.array([[1, 1], [2, 3], [3, 2], [4, 4.0]])
Xc, mu = center(X)
print("mean", mu, "\ncentred\n", Xc, "\ncovariance\n", covariance(Xc))
for s in ["eig", "svd", "power"]:
    p = PCA(2, s).fit(X)
    print(f"{s:5s} eigenvalues {p.explained_variance_}  ratio {p.explained_variance_ratio_}  PC1 {p.components_[0]}")
print("singular values", np.linalg.svd(Xc, compute_uv=False), " s^2/(n-1) =", np.linalg.svd(Xc, compute_uv=False) ** 2 / 3)
p = PCA(1).fit(X)
print("scores on PC1", p.transform(X).ravel(), " recon error (k=1):", reconstruction_error(X, 1), " = lambda_2 = 1/3")

print("\n== 2. Three solvers on a random 200x6 dataset ==")
rng = np.random.default_rng(1)
Xr = rng.standard_normal((200, 6)) @ rng.standard_normal((6, 6)) + 3
Xrc, _ = center(Xr)
ve, ce = pca_eig(Xrc); vs, cs = pca_svd(Xrc); vp, cp = pca_power(Xrc, 6)
print("eigh  ", ve)
print("svd   ", vs)
print("power ", vp)
print("max |eig-svd| =", np.abs(ve - vs).max(), "  max |eig-power| =", np.abs(ve - vp).max())
print("reconstruction error k=1..5 :", [round(reconstruction_error(Xr, k), 4) for k in range(1, 6)])
print("sum discarded eigenvalues   :", [round(ve[k:].sum(), 4) for k in range(1, 6)])

print("\n== 3. Synthetic glyphs (n=300, 144 pixels, true rank 5) ==")
G = make_glyph_data()
vals, _ = pca_svd(center(G)[0])
r = np.cumsum(vals) / vals.sum()
print("first 7 eigenvalues", vals[:7])
print("cumulative ratio k=1,3,5,10,30:", r[[0, 2, 4, 9, 29]])
print("k for 90%/95%:", choose_k(vals, 0.90), choose_k(vals, 0.95))
print("noise-floor eigenvalue (sigma^2 = 0.15^2 = 0.0225); mean of eigenvalues 6..144 =", vals[5:].mean())
for k in [1, 3, 5, 10, 30]:
    print(f"  k={k:2d} mean sq error per sample {reconstruction_error(G, k):.4f}")

print("\n== 4. Standardisation (height cm, weight g) ==")
rng = np.random.default_rng(0); h = rng.standard_normal(200)
Xu = np.c_[170 + 8 * h + 2 * rng.standard_normal(200), (70 + 10 * h + 4 * rng.standard_normal(200)) * 1000]
Z, _, _ = standardize(Xu)
print("raw  PC1", PCA(1).fit(Xu).components_[0], "ratio", PCA(1).fit(Xu).explained_variance_ratio_)
print("std  PC1", PCA(1).fit(Z).components_[0], "ratio", PCA(1).fit(Z).explained_variance_ratio_)

print("\n== 5. Whitening ==")
Zw = PCA(6, whiten=True).fit_transform(Xr)
print("cov of whitened data = I ?", np.allclose(np.cov(Zw.T), np.eye(6)))

print("\n== 6. Probabilistic PCA (k=5 on glyphs) ==")
m = ppca_fit(G, 5)
print("sigma2_ML =", m["sigma2"], " (true noise var 0.0225)  loglik", ppca_loglik(m, G))
print("\n== 7. Exercise checks ==")
C2 = np.array([[5 / 3, 4 / 3], [4 / 3, 5 / 3]])
v = np.array([1.0, 0.0])
for t in range(1, 5):
    v = C2 @ v; v /= np.linalg.norm(v)
    print(f"  power iter t={t}: tan(angle to PC1) = {abs((v[0]-v[1])/(v[0]+v[1])):.3e}   predicted (1/9)^t = {(1/9)**t:.3e}")
mm = ppca_fit(X, 1)
print("  PPCA on worked example k=1: sigma2 =", mm["sigma2"], " |w| =", np.linalg.norm(mm["W"]), " sqrt(8/3) =", np.sqrt(8 / 3))
Xw = np.random.default_rng(5).standard_normal((4, 100))
print("  n=4, d=100: number of eigenvalues > 1e-10 :", int((pca_eig(center(Xw)[0])[0] > 1e-10).sum()))
print("  Ex: C=[[2,1],[1,2]] eigenvalues", np.linalg.eigvalsh(np.array([[2, 1], [1, 2.0]])))
print(f"\nelapsed {time.time() - t0:.1f}s")
