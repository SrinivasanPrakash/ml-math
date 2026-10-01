"""Neural network from scratch (NumPy only).

Conventions (see README.md, "Notation"):
    * A batch is a matrix with one *row* per sample: X has shape (N, d).
    * Dense layer: W has shape (n_in, n_out), b has shape (n_out,), so
      Z = A_prev @ W + b has shape (N, n_out).                     README Eq. (E4)
    * Every loss is the *mean* over the N samples of the batch, so
      the 1/N factor lives in the loss gradient and nowhere else.

Every layer implements
    forward(X)  -> output, caching what backward needs
    backward(G) -> dL/dX given G = dL/d(output); also fills self.grads
so a network is just the chain rule applied layer by layer (README Sec. 7).
"""

import numpy as np


# --------------------------------------------------------------------------
# Initialisation  (README Sec. 8)
# --------------------------------------------------------------------------
def init_weights(n_in, n_out, scheme, rng):
    """Return a (n_in, n_out) weight matrix.

    scheme:
        'xavier'  Var(w) = 2/(n_in+n_out)   README Eq. (E17)
        'he'      Var(w) = 2/n_in           README Eq. (E18)
        'normal'  Var(w) = 1                (deliberately too large)
        'small'   Var(w) = 1e-4             (deliberately too small)
    """
    if scheme == "xavier":
        std = np.sqrt(2.0 / (n_in + n_out))
    elif scheme == "he":
        std = np.sqrt(2.0 / n_in)
    elif scheme == "normal":
        std = 1.0
    elif scheme == "small":
        std = 0.01
    else:
        raise ValueError(f"unknown init scheme {scheme!r}")
    return rng.standard_normal((n_in, n_out)) * std


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
class Dense:
    """Affine layer  Z = A W + b.   README Eq. (E4), gradients (E15), (E16)."""

    def __init__(self, n_in, n_out, init="xavier", rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.W = init_weights(n_in, n_out, init, rng)
        self.b = np.zeros(n_out)
        self.grads = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}

    def forward(self, X):
        self.X = X                      # cache A_{l-1}, shape (N, n_in)
        return X @ self.W + self.b      # (N, n_in)(n_in, n_out) -> (N, n_out)

    def backward(self, G):
        # G = Delta_l = dL/dZ_l, shape (N, n_out)
        self.grads["W"] = self.X.T @ G          # Eq. (E15): (n_in,N)(N,n_out)
        self.grads["b"] = G.sum(axis=0)         # Eq. (E16): 1^T Delta
        return G @ self.W.T                     # dL/dA_{l-1}, used in Eq. (E14)

    def parameters(self):
        return [(self.W, self.grads["W"]), (self.b, self.grads["b"])]


class _Elementwise:
    """Base for activations A = f(Z); backward is G * f'(Z)  (Eq. E14)."""

    def parameters(self):
        return []

    def backward(self, G):
        return G * self.deriv()


class Sigmoid(_Elementwise):
    """sigma(z) = 1/(1+e^-z), sigma' = sigma (1 - sigma).   Eq. (E5)"""

    def forward(self, Z):
        # numerically stable: never exponentiate a large positive number
        out = np.empty_like(Z)
        pos = Z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
        e = np.exp(Z[~pos])
        out[~pos] = e / (1.0 + e)
        self.out = out
        return out

    def deriv(self):
        return self.out * (1.0 - self.out)


class Tanh(_Elementwise):
    """tanh' = 1 - tanh^2.   Eq. (E6)"""

    def forward(self, Z):
        self.out = np.tanh(Z)
        return self.out

    def deriv(self):
        return 1.0 - self.out ** 2


class ReLU(_Elementwise):
    """max(0, z), derivative 1[z > 0].   Eq. (E7)"""

    def forward(self, Z):
        self.mask = Z > 0
        return Z * self.mask

    def deriv(self):
        return self.mask.astype(float)


class Softmax:
    """Row-wise softmax, Eq. (E9). Backward uses the Jacobian of Eq. (E11):

        dL/dz_j = p_j ( g_j - sum_i g_i p_i )     (row-wise)

    Normally you do NOT use this layer for training: SoftmaxCrossEntropy
    fuses it with the loss and gives the much simpler (p - y)/N, Eq. (E12).
    """

    def parameters(self):
        return []

    def forward(self, Z):
        self.p = softmax(Z)
        return self.p

    def backward(self, G):
        inner = (G * self.p).sum(axis=1, keepdims=True)
        return self.p * (G - inner)


def softmax(Z):
    """Stable softmax (subtract the row max; the result is unchanged)."""
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Losses.  forward(pred, target) -> scalar, backward() -> dL/dpred
# --------------------------------------------------------------------------
def one_hot(y, k):
    y = np.asarray(y)
    if y.ndim == 2:
        return y.astype(float)
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y.astype(int)] = 1.0
    return out


class MSE:
    """L = 1/(2N) sum ||a - y||^2,   dL/da = (a - y)/N.   Eq. (E8)"""

    def forward(self, pred, target):
        self.diff = pred - target
        self.N = pred.shape[0]
        return 0.5 * np.sum(self.diff ** 2) / self.N

    def backward(self):
        return self.diff / self.N


class SoftmaxCrossEntropy:
    """Fused softmax + cross-entropy on logits Z.   Eq. (E10), (E12)

    L = -1/N sum_n log p_{n, y_n},    dL/dZ = (P - Y)/N.
    target: integer labels (N,) or one-hot (N, K).
    """

    def forward(self, Z, target):
        self.P = softmax(Z)
        self.Y = one_hot(target, Z.shape[1])
        self.N = Z.shape[0]
        # log-softmax computed stably, never log(0)
        Zs = Z - Z.max(axis=1, keepdims=True)
        logp = Zs - np.log(np.exp(Zs).sum(axis=1, keepdims=True))
        return -np.sum(self.Y * logp) / self.N

    def backward(self):
        return (self.P - self.Y) / self.N


class CrossEntropy:
    """Cross-entropy on *probabilities* P (pair with the Softmax layer).

    L = -1/N sum Y * log P,   dL/dP = -(Y / P)/N.   Eq. (E10)
    """

    def forward(self, P, target):
        self.P = P
        self.Y = one_hot(target, P.shape[1])
        self.N = P.shape[0]
        return -np.sum(self.Y * np.log(P + 1e-300)) / self.N

    def backward(self):
        return -(self.Y / (self.P + 1e-300)) / self.N


class BCEWithLogits:
    """Binary cross-entropy on a logit z (N,1) or (N,): Eq. (E2), (E3).

    L = 1/N sum [ softplus(z) - y z ],   dL/dz = (sigma(z) - y)/N.
    """

    def forward(self, Z, target):
        self.shape = Z.shape
        self.Z = Z.reshape(-1)
        self.y = np.asarray(target, dtype=float).reshape(-1)
        self.N = self.Z.shape[0]
        softplus = np.maximum(self.Z, 0) + np.log1p(np.exp(-np.abs(self.Z)))
        return np.sum(softplus - self.y * self.Z) / self.N

    def backward(self):
        s = 1.0 / (1.0 + np.exp(-self.Z))
        return ((s - self.y) / self.N).reshape(self.shape)


# --------------------------------------------------------------------------
# Network container
# --------------------------------------------------------------------------
class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, G):
        for layer in reversed(self.layers):
            G = layer.backward(G)
        return G

    def parameters(self):
        out = []
        for layer in self.layers:
            out.extend(layer.parameters())
        return out

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def mlp(sizes, activation="relu", init=None, rng=None):
    """Build Dense/activation stack; the last Dense has no activation.

    sizes = [d_in, h1, h2, ..., d_out]
    """
    rng = np.random.default_rng() if rng is None else rng
    act = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}[activation]
    if init is None:
        init = "he" if activation == "relu" else "xavier"
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Dense(sizes[i], sizes[i + 1], init=init, rng=rng))
        if i < len(sizes) - 2:
            layers.append(act())
    return Sequential(layers)


# --------------------------------------------------------------------------
# Optimisers (README Sec. 10).  step() updates parameters in place.
# --------------------------------------------------------------------------
class SGD:
    """theta <- theta - lr * g.   Eq. (E19)"""

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params):
        for p, g in params:
            p -= self.lr * g


class Momentum:
    """v <- beta v + g ;  theta <- theta - lr v.   Eq. (E20)"""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, params):
        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        for v, (p, g) in zip(self.v, params):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    """Adam with bias correction.   Eq. (E21), (E22)"""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params):
        if self.m is None:
            self.m = [np.zeros_like(p) for p, _ in params]
            self.v = [np.zeros_like(p) for p, _ in params]
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t          # bias corrections, Eq. (E22)
        c2 = 1.0 - self.b2 ** self.t
        for m, v, (p, g) in zip(self.m, self.v, params):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------
def fit(model, loss, opt, X, y, epochs=100, batch_size=None, rng=None,
        callback=None):
    """Mini-batch training. batch_size=None means full batch (deterministic).

    Returns the list of per-epoch mean training losses (mean over the batches
    seen in that epoch, weighted by batch size). callback(epoch, model) is
    called *before* epoch 0 (untrained) and after every epoch.
    """
    N = X.shape[0]
    bs = N if batch_size is None else batch_size
    rng = np.random.default_rng(0) if rng is None else rng
    history = []
    if callback:
        callback(0, model)
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(N) if bs < N else np.arange(N)
        total = 0.0
        for s in range(0, N, bs):
            b = idx[s:s + bs]
            out = model.forward(X[b])
            total += loss.forward(out, y[b]) * len(b)
            model.backward(loss.backward())
            opt.step(model.parameters())
        history.append(total / N)
        if callback:
            callback(epoch, model)
    return history


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def make_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def make_moons(n=200, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = np.linspace(0, np.pi, n1)
    t2 = np.linspace(0, np.pi, n - n1)
    A = np.c_[np.cos(t1), np.sin(t1)]
    B = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([A, B]) + noise * rng.standard_normal((n, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n - n1, int)]
    return X, y


def make_spirals(n_per_class=100, classes=3, noise=0.2, seed=0):
    """K interleaved spiral arms (angle noise `noise`)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * classes, 2))
    y = np.zeros(n_per_class * classes, dtype=int)
    for k in range(classes):
        ix = slice(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = (np.linspace(k * 4, (k + 1) * 4, n_per_class)
             + noise * rng.standard_normal(n_per_class))
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def accuracy(model, X, y):
    return float(np.mean(model.predict(X) == y))
