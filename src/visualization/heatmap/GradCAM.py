import numpy as np
import time


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):

        self.activations = output.detach()
        # output.retain_grad()

    def save_gradient(self, module, grad_input, grad_output):
        # print("backward hook called")
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_index):
        self.model.zero_grad()
        output = self.model(input_tensor)

        loss = output[:, class_index].sum()

        loss.backward()

        assert (
            self.gradients is not None
        ), "Hook never called: bad layer or requires_grad=False"

        # Compute the weights
        pooled_gradients = self.gradients.mean(dim=[0, 2, 3])  # global average pooling
        activations = self.activations[0]  # remove batch dimension
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = activations.sum(dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-8  # normalize
        # clear stored data to free CPU memory
        self.activations = None
        self.gradients = None

        return heatmap
