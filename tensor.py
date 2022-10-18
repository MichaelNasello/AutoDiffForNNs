"""
Provides a PyTorch-like implementation of a `Tensor` that stores a high-dimensional array, along with a complete history
of how it was produced, and the necessary methods required to perform backpropagation through that history and compute
gradients of any trainable items.

A `Tensor` consists of:
 - v: n-dimensional tensor
 - grad_fn: method to propagate error from self to self.parents
 - parents: [list of `Tensor`s]

The implementation of `Tensor` allows easy, pythonic operations of tensors with Auto-diff working in the background.
Consider the following example with:

    x: [batch, in_features]
    w: [in_features, out_features]
    b: [out_features]
    y: [batch, out_features]

We can compute model outputs as:

    hidden = x @ W1 + b1
    y_hat = hidden @ W2 + b2

The MSE loss is simply:

    loss = y_hat.mse_loss(y)

At this point, the `loss` variable contains a complete history of relevant operations, stored recursively in
loss.parents. To perform backpropagation, we simply call:

    loss.backward(
        del_loss_del_out_upstream=Tensor(np.array([1])),
        grad_fn=loss.mse_backwards,
        from_loss=True
    )

and error is recursively propagated until reaching the leafs of the tree (input of the network and parameters).
"""

import copy
import numpy as np


class Tensor:
    def __init__(self, value, name=None, grad_fn=None, requires_grad=True, is_param=False, parents=None, inputs=None):
        assert type(value) in [np.array, np.ndarray], f'type(value) is {type(value)}'

        self.v = value
        self.name = name
        self.grad_fn = grad_fn
        self.requires_grad = requires_grad
        self.is_param = is_param

        self.shape = self.v.shape
        self.parents = parents

        if self.requires_grad:
            self.input = inputs

        if self.is_param:
            self.grad = np.zeros_like(self.v)
        else:
            self.grad = None

    def __matmul__(self, tensor):
        # Perform matmul operation
        self.input = tensor
        if self.shape == (1,) or tensor.shape == (1,):
            out = self.v * tensor.v
        else:
            out = self.v @ tensor.v

        # If inclusion in history required, create Tensor with corresponding grad_fn, with reference to parents
        if self.requires_grad or tensor.requires_grad:
            return Tensor(
                out,
                grad_fn=self.matmul_backwards,
                parents=[self, tensor],
                inputs=tensor
            )
        else:
            return Tensor(out)

    def __add__(self, tensor):
        self.input = tensor

        # If inclusion in history required, create Tensor with corresponding grad_fn, with reference to parents
        if self.requires_grad or tensor.requires_grad:
            return Tensor(
                self.v + tensor.v,  # Perform addition operation
                grad_fn=self.add_backwards,
                parents=[self, tensor],
                inputs=tensor
            )
        else:
            return Tensor(self.v + tensor.v)

    def __sub__(self, tensor):
        # If inclusion in history required, create Tensor with corresponding grad_fn, with reference to parents
        if self.requires_grad or tensor.requires_grad:
            return Tensor(
                self.v - tensor.v,  # Perform subtraction operation
                grad_fn=self.sub_backwards,
                parents=[self, tensor],
                inputs=tensor
            )
        else:
            return Tensor(self.v - tensor.v)

    def relu(self):
        # If inclusion in history required, create Tensor with corresponding grad_fn, with reference to parents
        if self.requires_grad:
            return Tensor(
                np.maximum(0, self.v),  # Perform ReLU operation
                grad_fn=self.relu_backwards,
                parents=[self],
                inputs=self
            )
        else:
            return Tensor(np.maximum(0, self.v))

    def mse_loss(self, y):
        # Track y_hat and y
        self.input = [self.v, y.v]

        # Perform MSE operation
        mse_loss = ((self.v - y.v)**2) / 2

        # Return Tensor with mean (across batch and output feature dimension) scalar loss
        return Tensor(
            np.array([mse_loss.mean()]),
            grad_fn=self.grad_fn,
            parents=self.parents,
            inputs=[self, y]
        )

    def bce_loss(self, y):
        # Track y_hat and y
        self.input = [self.v, y.v]

        # Perform BCE operation
        bce_loss = -(np.multiply(y.v, np.log(self.v)) + np.multiply(1 - y.v, np.log(1 - y.v)))

        # Return Tensor with mean (across batch) scalar loss
        return Tensor(
            np.array([bce_loss.mean()]),
            grad_fn=self.grad_fn,
            parents=self.parents,
            inputs=[self, y]
        )

    def transpose(self):
        transposed = copy.deepcopy(self)
        transposed.v = np.transpose(transposed.v)

        self.shape = transposed.v.shape
        return transposed

    def matmul_backwards(self, del_loss_del_out_upstream):
        del_loss_del_in = del_loss_del_out_upstream @ self.input.transpose()  # dL/dx = dz/dy * w^t
        del_out_del_param = self.transpose()  # dz / dW = x^t

        del_loss_del_param = del_out_del_param @ del_loss_del_out_upstream  # dL/dW = dz/dW @ dL/dz
        return del_loss_del_in, del_loss_del_param

    @staticmethod
    def add_backwards(del_loss_del_out_upstream):
        del_loss_del_in = del_loss_del_out_upstream  # dL/dx = dL/dz
        del_out_del_param = Tensor(np.array([1]))  # dz/db = 1

        del_loss_del_param = Tensor(np.sum(del_loss_del_out_upstream.v, 0)) @ del_out_del_param  # dL/db = dL/dz @ dz/db
        return del_loss_del_in, del_loss_del_param

    def sub_backwards(self):
        # Not required
        pass

    def relu_backwards(self, del_loss_del_out_upstream):
        # dL/dx = max(0, x) * dL/dz
        del_loss_del_in = Tensor(
            (self.v > 0) * del_loss_del_out_upstream.v
        )
        return del_loss_del_in

    def mse_backwards(self):
        input_shape = self.input[0].shape
        batch_size = input_shape[0] * input_shape[1]

        # dL/dx = (y_hat - y) * (2 / batch_size)
        del_loss_del_in = Tensor(
            (self.input[0] - self.input[1]).v * np.array([1 / batch_size])
        )

        return del_loss_del_in

    def bce_backwards(self):
        # dL/dx = (y_hat - y) * (2 / batch_size)
        del_loss_del_in = Tensor(
            (self.input[0] - self.input[1]).v * np.array([2])
        )

        return del_loss_del_in

    def backwards(self, del_loss_del_out_upstream, grad_fn, from_loss=False):
        if from_loss:
            # Get dL/dx
            del_loss_del_in = grad_fn()
        elif grad_fn == self.relu_backwards:
            # Get dL/dx, passing in dL/dz
            del_loss_del_in = grad_fn(del_loss_del_out_upstream)
        else:
            # Get dL/dx, dL/d_theta, passing in dL/dz
            del_loss_del_in, del_loss_del_param = grad_fn(del_loss_del_out_upstream)

        # Updating param if self is trainable
        if self.is_param:
            self.grad = del_loss_del_param

        # Backpropagate error if self if dependant on parent `Tensor`s
        if self.parents:
            for parent in self.parents:
                if parent.requires_grad:
                    parent.backwards(
                        del_loss_del_out_upstream=del_loss_del_in,  # Pass dL/dx as dL/dz in next operation
                        grad_fn=self.grad_fn  # Pass grad_fn necessary to compute parents dL/d_theta, dL/dx
                    )

    def __repr__(self):
        return str(
            {
                'name': self.name,
                'shape': self.shape
            }
        )
