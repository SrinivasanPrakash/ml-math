"""Hand-worked 2-2-2-1 example (README Sec. 13), reproduced with the library."""
import numpy as np

import nn


def run():
    x = np.array([[1.0, 2.0]])
    y = np.array([[1.0]])
    d1 = nn.Dense(2, 2)
    d2 = nn.Dense(2, 2)
    d3 = nn.Dense(2, 1)
    d1.W = np.array([[1.0, -1.0], [0.5, 1.0]])
    d2.W = np.array([[0.5, 1.0], [-0.5, 0.5]])
    d3.W = np.array([[1.0], [-0.5]])
    r1, r2 = nn.ReLU(), nn.ReLU()
    net = nn.Sequential([d1, r1, d2, r2, d3])
    loss = nn.MSE()
    z3 = net.forward(x)
    L = loss.forward(z3, y)
    net.backward(loss.backward())
    # deltas dL/dZ_l are what flows into each Dense backward; recover them
    delta3 = loss.backward()
    delta2 = (delta3 @ d3.W.T) * r2.deriv()
    delta1 = (delta2 @ d2.W.T) * r1.deriv()
    return {"z1": x @ d1.W, "z2": r1.forward(x @ d1.W) @ d2.W,
            "z3": z3[0, 0], "loss": L, "delta3": delta3, "delta2": delta2[0],
            "delta1": delta1[0], "dW1": d1.grads["W"], "dW2": d2.grads["W"],
            "dW3": d3.grads["W"], "db1": d1.grads["b"], "db2": d2.grads["b"],
            "db3": d3.grads["b"], "a1": r1.forward(x @ d1.W),
            "a2": r2.forward(r1.forward(x @ d1.W) @ d2.W)}


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    for k, v in run().items():
        print(k, v)
