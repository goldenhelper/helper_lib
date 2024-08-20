import torch, copy
import torch.optim as optim
from matplotlib import pyplot as plt
import math
from collections import OrderedDict
from IPython.display import clear_output
from torchvision.transforms import functional as TF


def save_layer_shapes(model, input_shape=None, get_output_shape=None):
    """
    Create a deep copy of the model, run it on the 'meta' device to determine
    the input and output shapes of each layer, and store these shapes as attributes
    in the original model.
    The model should have its shape stored in an attribute input_shape
    or it should be explitly passed into this function.

    Args:
        model (torch.nn.Module): The original model for which input/output shapes are determined.

    Returns:
        dict: A dictionary with layer names as keys and tuples of input/output shapes as values.
    """
    if input_shape is None:
        if hasattr(model, 'input_shape'):
            input_shape = model.input_shape
        else:
            raise ValueError('The model does not have an input shape attribute nor it was passed into the function.')

    with torch.device("meta"):
        model_on_meta = copy.deepcopy(model).to('meta')
        inp = torch.randn(input_shape, device='meta')

        def fw_hook(module, input, output):
            if len(input) == 1:
                module.input_shape = input[0].shape # input[0] because pytorch returns a tuple for the input
            else:
                print("Warning. Non-standard input shape. Input shape doesn't get stored.")

            if hasattr(output, 'shape'):
                module.output_shape = output.shape
            else:
                module.output_shape = get_output_shape(output)

        # Register hooks to capture input/output shapes
        hook_handles = []
        for name, layer in model_on_meta.named_modules():
            handle = layer.register_forward_hook(fw_hook)
            hook_handles.append(handle)

        # Perform a forward pass to trigger the hooks
        model_on_meta(inp)

    # Copy the input/output shapes back to the original model's layers
    for name, layer in model.named_modules():
        if name != '':
            if hasattr(model_on_meta._modules[name], 'output_shape'):
                layer.input_shape = model_on_meta._modules[name].input_shape
                layer.output_shape = model_on_meta._modules[name].output_shape

    return {name: (layer.input_shape, layer.output_shape) for name, layer in model.named_modules() if
            hasattr(layer, 'output_shape')}


def activations_from_whole_dataset(model, loader, device, output_device, compute_labels=False):
    """
    Compute activations of the last layer for the entire dataset using the provided data loader.

    Args:
        model (torch.nn.Module): The model from which activations are computed.
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        device (torch.device): Device on which the model and data reside.
        output_device (torch.device): Device where the activations are stored.
        compute_labels (bool): Whether to also return labels with activations.

    Returns:
        torch.Tensor: Activations of the last layer.
        (torch.Tensor, optional): Corresponding labels if compute_labels is True.
    """
    model.eval()
    activations = []
    labels = []
    with torch.no_grad():
        for data, targets in loader:
            output = model(data.to(device))
            activations.append(output.to(output_device))
            if compute_labels:
                labels.append(targets)
    if compute_labels:
        return torch.cat(activations), torch.cat(labels)
    else:
        return torch.cat(activations)


def visualize(nums, dataset):
    """
    Visualize a set of images from a dataset given their indices.

    Args:
        nums (list): List of indices of images to visualize.
        dataset (torch.utils.data.Dataset): Dataset containing the images.
    """
    figure = plt.figure(figsize=(8, 8))
    side_length = math.ceil(math.sqrt(len(nums)))
    for i in range(len(nums)):
        img = dataset[nums[i]][0].squeeze()
        label = dataset[nums[i]][1]
        figure.add_subplot(side_length, side_length, i + 1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def visualize_ims(images, channel, text=None):
    """
    Visualize a batch of images along a specific channel.

    Args:
        images (torch.Tensor): A batch of images to visualize.
        channel (int): The channel of the images to visualize.
        text (str, optional): Optional title for the visualization.
    """
    images = images.detach()
    figure = plt.figure(figsize=(8, 8))
    side_length = math.ceil(math.sqrt(len(images)))
    for i in range(len(images)):
        img = images[i, channel].cpu()
        figure.add_subplot(side_length, side_length, i + 1)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    if text:
        plt.title(text)
    plt.show()


def store_activations(module):
    """
    Create a hook to store output activations of a module.

    Args:
        module (torch.nn.Module): The module to attach the hook to.

    Returns:
        torch.utils.hooks.RemovableHandle: Handle for the hook that can be used to remove it.
    """

    def storing_activations_hook(module, input, output):
        module.stored_activations = output

    module.stored_activations = torch.zeros(module.output_shape)
    return module.register_forward_hook(storing_activations_hook)


def store_inputs(module):
    """
    Create a hook to store input activations of a module.

    Args:
        module (torch.nn.Module): The module to attach the hook to.

    Returns:
        torch.utils.hooks.RemovableHandle: Handle for the hook that can be used to remove it.
    """

    def storing_input_activations_hook(module, input, output):
        if len(input) != 1:
            print("Warning: Non-standard input shape.")
        module.stored_inputs = input[0]

    module.stored_inputs = torch.zeros(module.input_shape)
    return module.register_forward_hook(storing_input_activations_hook)


def print_registered_forward_hooks(model, model_name=None):
    """
    Print all registered forward hooks in the model.

    Args:
        model (torch.nn.Module): The model whose hooks are to be printed.
        model_name (str, optional): The name of the model. If not provided, the model's class name is used.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    print(f"Registered forward hooks of {model_name}:")
    for module in model.children():
        hooks = module._forward_hooks
        if hooks:
            for hook_id, hook in hooks.items():
                print(f"Module: {module}, Hook ID: {hook_id}, Hook: {hook.__qualname__}")


def remove_all_forward_hooks(model):
    """
    Remove all forward hooks from the model.

    Args:
        model (torch.nn.Module): The model from which to remove all forward hooks.
    """
    for child in model.children():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            remove_all_forward_hooks(child)


def optim_inputs_for_neurons_in_layer(model, layer, device, mode='activations', neuron_indices=None, init_inputs=None,
                                      alpha=20.0, image_sparsity=50.0, num_steps=1000):
    """
    Optimize input images for specific neurons in a given layer.

    Args:
        model (torch.nn.Module): The model containing the target layer.
        layer (torch.nn.Module): The layer for which neurons are to be optimized.
        device (torch.device): The device to run the optimization on.
        mode (str): Whether to optimize based on 'activations' or 'inputs'.
        neuron_indices (torch.Tensor): Indices of the neurons to optimize for.
        init_inputs (str or torch.Tensor): Initialization strategy or tensor for the inputs.
        alpha (float): Weighting factor for the target neuron activations in the loss.
        image_sparsity (float): Regularization term to enforce sparsity of the optimized inputs.
        num_steps (int): Number of optimization steps.

    Returns:
        torch.Tensor: The optimized input images.
    """
    model_on_device = model.to(device)
    layer_on_device = layer.to(device)
    neuron_indices = neuron_indices.to(device)

    if init_inputs == 'random':
        optimized_inputs = torch.randn((len(neuron_indices), *model.input_shape[1:]), device=device, requires_grad=True)
    elif init_inputs == 'zeros':
        optimized_inputs = torch.zeros((len(neuron_indices), *model.input_shape[1:]), device=device, requires_grad=True)
    else:
        optimized_inputs = init_inputs
        optimized_inputs.requires_grad_(True)

    optimizer = optim.Adam([optimized_inputs], lr=1e-3)

    def get_activations():
        if mode == 'activations':
            return layer_on_device.stored_activations
        elif mode == 'inputs':
            return layer_on_device.stored_inputs
        else:
            raise ValueError("Mode must be 'activations' or 'inputs'")

    rotation_angles = [5, 10, 15]

    activations_to_optimize_for = get_activations()
    print(f"Initial activations: {get_activations()}")

    if max(neuron_indices) > activations_to_optimize_for.shape[-1]:
        print("Warning: neuron_indices out of range.")

    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        model_on_device(optimized_inputs)
        activations_to_optimize_for = get_activations()
        batch_of_activations = activations_to_optimize_for[:len(neuron_indices)]

        target_neurons_activations = torch.sum(batch_of_activations[torch.arange(len(neuron_indices)), neuron_indices])
        original_loss = torch.sum(batch_of_activations) - alpha * target_neurons_activations
        total_loss = original_loss + image_sparsity * torch.sum(optimized_inputs)

        for angle in rotation_angles:
            rotated_images = TF.rotate(optimized_inputs, angle)
            model_on_device(rotated_images)
            rotated_activations = get_activations()
            batch_of_rotated_activations = rotated_activations[:len(neuron_indices)]
            target_neurons_activations = torch.sum(
                batch_of_rotated_activations[torch.arange(len(neuron_indices)), neuron_indices])
            rotation_loss = torch.sum(batch_of_rotated_activations) - alpha * target_neurons_activations
            total_loss += rotation_loss + image_sparsity * torch.sum(rotated_images)

        total_loss.backward()
        optimizer.step()

        if step % (num_steps // 10) == 0:
            print(f"step#{step + 1}")
            print(f"Original loss: {original_loss.item()}, Rotation loss: {rotation_loss.item()}, Sparsity loss: {image_sparsity * torch.sum(optimized_inputs).item()}")
            print(f"Activations: {get_activations()}")

    return optimized_inputs.detach()


def bar_diagram(activations_to_vis, figsize):
    """
    Plot a bar diagram of the given activations.

    Args:
        activations_to_vis (torch.Tensor): Tensor containing activations to visualize.
        figsize (tuple): Size of the figure.
    """
    plt.figure(figsize=figsize)
    print(activations_to_vis)
    bar_width = 0.4
    indices = torch.arange(len(activations_to_vis))

    plt.bar(indices, activations_to_vis.cpu(), width=bar_width, color='darkmagenta', alpha=0.7, label='Values')
    plt.xticks(indices)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5, axis='y')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Sparse representation')
    plt.tight_layout()
    plt.show()

def get_device(model):
    return next(model.parameters()).device