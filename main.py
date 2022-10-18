"""
Performs all tasks outlined in Assignment Doc.
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np

from tensor import Tensor


class ModelWithAutoDiff:
    def __init__(self, path_to_parameters=None):
        # If parameters are specified
        if path_to_parameters is not None:
            with open(path_to_parameters, 'rb') as file:
                parameters = pickle.load(file)

                # Set parameters with type Tensor
                self.w1 = Tensor(np.transpose(parameters['w1']), name='w1', is_param=True)
                self.b1 = Tensor(parameters['b1'], name='b1', is_param=True)

                self.w2 = Tensor(np.transpose(parameters['w2']), name='w2', is_param=True)
                self.b2 = Tensor(parameters['b2'], name='b2', is_param=True)

                self.w3 = Tensor(np.transpose(parameters['w3']), name='w3', is_param=True)
                self.b3 = Tensor(parameters['b3'], name='b3', is_param=True)
        else:
            # Instantiate weights of type Tensor with random values
            self.w1 = Tensor(
                np.random.uniform(-1, 1, (2, 10)),
                name='w1',
                is_param=True
            )
            self.b1 = Tensor(
                np.random.uniform(-1, 1, 10),
                name='b1',
                is_param=True
            )

            self.w2 = Tensor(
                np.random.uniform(-1, 1, (10, 10)),
                name='w2',
                is_param=True
            )
            self.b2 = Tensor(
                np.random.uniform(-1, 1, 10),
                name='b2',
                is_param=True
            )

            self.w3 = Tensor(
                np.random.uniform(-1, 1, (10, 1)),
                name='w3',
                is_param=True
            )
            self.b3 = Tensor(
                np.random.uniform(-1, 1, 1),
                name='b3',
                is_param=True
            )

        self.parameters = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]

    def __call__(self, x):
        # Forward pass of network
        hidden1 = ((x @ self.w1) + self.b1).relu()
        hidden2 = ((hidden1 @ self.w2) + self.b2).relu()
        hidden3 = (hidden2 @ self.w3) + self.b3

        return hidden3

    def train(self, x, y, n_epochs, lr):
        assert len(x.shape) == 2
        assert len(y.shape) == 2

        losses = []
        for epoch in range(n_epochs):
            # Printing gradients for first layer on first input-output pair of dataset
            if epoch == 0:
                x_0 = Tensor(x.v[0][None], requires_grad=False)
                y_0 = Tensor(y.v[0][None])

                y_hat_0 = self(x_0)
                loss_0 = y_hat_0.mse_loss(y_0)
                loss_0.backwards(
                    del_loss_del_out_upstream=Tensor(np.array([1])),
                    grad_fn=loss_0.mse_backwards,
                    from_loss=True
                )
                self.print_gradients_for_first_layer()

            # Generate outputs and loss
            model_outputs = self(x)
            loss = model_outputs.mse_loss(y)

            # Perform backward pass, updating model parameter gradients
            loss.backwards(
                del_loss_del_out_upstream=Tensor(np.array([1])),
                grad_fn=loss.mse_backwards,
                from_loss=True
            )
            losses.append(loss.v)

            # Apply gradients
            self.update_parameters(lr)

        return losses

    def update_parameters(self, lr):
        # Although not implicitly stated, parameters updating in Minibatch-SGD fashion
        for parameter in self.parameters:
            parameter.v -= lr * parameter.grad.v

    def print_gradients_for_first_layer(self):
        print('Gradient of first layer weight after first input-target pair')
        print(self.w1.grad.v)

        print('Gradient of first layer bias after first input-target pair:')
        print(self.b1.grad.v)

    def print_gradients(self):
        print(f'Gradients for model:')
        for parameter in self.parameters:
            print(f'\t{parameter.name}: {parameter.grad}')

    @staticmethod
    def plot_losses(losses):
        plt.plot(losses, '*')

        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')

        plt.savefig('training_curve.png')
        plt.show()


if __name__ == '__main__':
    model = ModelWithAutoDiff('assignment-one-test-parameters.pkl')

    # Extracting inputs and targets from pickle file
    with open('assignment-one-test-parameters.pkl', 'rb') as file:
        test_values = pickle.load(file)

    inputs = Tensor(test_values['inputs'], requires_grad=False)
    targets = Tensor(np.expand_dims(test_values['targets'], axis=1), requires_grad=False)

    # Training hyperparameters
    num_epochs = 6
    learning_rate = 0.01

    # Train model
    loss_over_epochs = model.train(inputs, targets, num_epochs, learning_rate)

    # Plots loss before any training, and after [1, 2, 3, 4, and 5] epochs (6 diff losses)
    model.plot_losses(loss_over_epochs)
