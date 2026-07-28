import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.layers import Dense


def test_forward_output_shape():
    """Output forward harus punya shape (batch_size, output_dim)."""
    layer = Dense(input_dim=4, output_dim=3)
    X = np.random.randn(10, 4)
    out = layer.forward(X)
    assert out.shape == (10, 3), f"Expected shape (10, 3), got {out.shape}"


def test_forward_computation():
    """Forward harus menghasilkan X @ W + b secara matematis benar."""
    layer = Dense(input_dim=2, output_dim=1)
    layer.W = np.array([[2.0], [3.0]])
    layer.b = np.array([[1.0]])

    X = np.array([[1.0, 1.0], [2.0, 0.0]])
    out = layer.forward(X)

    expected = np.array([[6.0], [5.0]])  # (1*2+1*3+1), (2*2+0*3+1)
    np.testing.assert_allclose(out, expected)


def test_backward_output_shape():
    """dX dari backward harus sama shape-nya dengan X input."""
    layer = Dense(input_dim=4, output_dim=3)
    X = np.random.randn(5, 4)
    layer.forward(X)

    dZ = np.random.randn(5, 3)
    dX = layer.backward(dZ, learning_rate=0.01)

    assert dX.shape == X.shape, f"Expected dX shape {X.shape}, got {dX.shape}"


def test_backward_updates_weights():
    """Weight dan bias harus berubah setelah backward (bukti gradient descent jalan)."""
    layer = Dense(input_dim=3, output_dim=2)
    W_before = layer.W.copy()
    b_before = layer.b.copy()

    X = np.random.randn(8, 3)
    layer.forward(X)

    dZ = np.random.randn(8, 2)
    layer.backward(dZ, learning_rate=0.1)

    assert not np.allclose(layer.W, W_before), "Weights should change after backward"
    assert not np.allclose(layer.b, b_before), "Bias should change after backward"


def test_gradient_numerical_check():
    """
    Numerical gradient checking: bandingkan dW hasil backward()
    dengan gradient hasil finite-difference approximation.
    Ini cara paling kuat untuk mendeteksi bug matematis di backward pass.
    """
    np.random.seed(0)
    layer = Dense(input_dim=3, output_dim=1)
    X = np.random.randn(4, 3)

    def loss_fn(W):
        layer.W = W
        out = layer.forward(X)
        return np.sum(out ** 2)  # loss sederhana: sum of squares

    W = layer.W.copy()
    out = layer.forward(X)
    dZ = 2 * out  # gradient dari loss sum(out**2) terhadap out
    layer.backward(dZ, learning_rate=0.0)  # learning_rate=0 supaya W tidak berubah
    analytical_dW = layer.dW

    epsilon = 1e-5
    numerical_dW = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += epsilon
            loss_plus = loss_fn(W_plus)

            W_minus = W.copy()
            W_minus[i, j] -= epsilon
            loss_minus = loss_fn(W_minus)

            numerical_dW[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

    layer.W = W  # restore

    np.testing.assert_allclose(analytical_dW, numerical_dW, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    test_forward_output_shape()
    test_forward_computation()
    test_backward_output_shape()
    test_backward_updates_weights()
    test_gradient_numerical_check()
    print("All tests passed!")