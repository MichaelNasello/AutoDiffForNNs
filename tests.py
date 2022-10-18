"""
Performs tests to ensure correct gradient computations.
"""

import numpy as np
import torch
from torch import nn

from main import ModelWithAutoDiff
from tensor import Tensor


def build_autodiff_model():
    # Build model with randomly initialized parameters
    model = ModelWithAutoDiff(path_to_parameters=None)
    return model


def build_pytorch_model_with_same_weights(auto_model):
    # Build pytorch model, and copy over randomly initialized weights from custom model
    model = torch.nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    # Weights need transpose due to difference in forward pass implementation (ordering of W and x)
    model[0].weight.data = torch.Tensor(np.transpose(auto_model.w1.v))
    model[0].bias.data = torch.Tensor(auto_model.b1.v)

    model[2].weight.data = torch.Tensor(np.transpose(auto_model.w2.v))
    model[2].bias.data = torch.Tensor(auto_model.b2.v)

    model[4].weight.data = torch.Tensor(np.transpose(auto_model.w3.v))
    model[4].bias.data = torch.Tensor(auto_model.b3.v)

    return model


def zero_grad_pytorch_model(pytorch_model):
    for parameter in pytorch_model.parameters():
        parameter.grad.zero_()


if __name__ == '__main__':
    autodiff_model = build_autodiff_model()
    torch_model = build_pytorch_model_with_same_weights(autodiff_model)

    # For 10 trials
    for i in range(10):
        # Random inputs and targets
        dummy_input = np.random.uniform(-10, 20, (200, 2))
        dummy_target = np.random.uniform(-10, 20, (200, 1))

        # Model outputs
        autodiff_out = autodiff_model(Tensor(dummy_input, requires_grad=False))
        torch_out = torch_model(torch.Tensor(dummy_input))

        # Assert model outputs are the same (check of forward pass)
        assert np.allclose(autodiff_out.v, torch_out.data, atol=1e-3, rtol=1e-5)

        # Backward pass of both models
        autodiff_loss = autodiff_out.mse_loss(Tensor(dummy_target))
        autodiff_loss.backwards(
            del_loss_del_out_upstream=Tensor(np.array([1])), grad_fn=autodiff_loss.mse_backwards, from_loss=True
        )

        torch_loss = nn.functional.mse_loss(torch_out, torch.Tensor(dummy_target))
        torch_loss.backward()

        # Assert all model parameter gradients are the same (check of backward pass)
        assert np.allclose(
            autodiff_model.w1.grad.v,
            np.transpose(torch_model[0].weight.grad / 2),  # Divide by two as our loss function is 1 / 2 of PyTorch's
            atol=1e-3,
            rtol=1e-5
        )
        assert np.allclose(
            autodiff_model.b1.grad.v,
            torch_model[0].bias.grad / 2,
            atol=1e-3,
            rtol=1e-5
        )

        assert np.allclose(
            autodiff_model.w2.grad.v,
            np.transpose(torch_model[2].weight.grad / 2),
            atol=1e-3,
            rtol=1e-5
        )
        assert np.allclose(
            autodiff_model.b2.grad.v,
            torch_model[2].bias.grad / 2,
            atol=1e-3,
            rtol=1e-5
        )

        assert np.allclose(
            autodiff_model.w3.grad.v,
            np.transpose(torch_model[4].weight.grad / 2),
            atol=1e-3,
            rtol=1e-5
        )
        assert np.allclose(
            autodiff_model.b3.grad.v,
            torch_model[4].bias.grad / 2,
            atol=1e-3,
            rtol=1e-5
        )

        zero_grad_pytorch_model(torch_model)

    print('All tests passed!')
