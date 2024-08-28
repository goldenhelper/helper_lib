import torch, copy
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.transforms import functional as TF
from matplotlib import pyplot as plt
import math
from collections import OrderedDict
from IPython.display import clear_output
import torch.nn.functional as F

def get_device(model):
    return next(model.parameters()).device

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

        self.input_shape = (1, input_size)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded

def get_module_by_name(model, name):
    """
    Recursively get a module within a model by its name.

    Args:
        model (torch.nn.Module): The model containing the module.
        name (str): The name of the module.

    Returns:
        torch.nn.Module: The module if found, else None.
    """
    if '.' not in name:
        return model._modules.get(name, None)
    else:
        name_list = name.split('.')
        return get_module_by_name(model._modules.get(name_list[0]), '.'.join(name_list[1:]))

def save_layer_shapes(model, input_shape=None, get_output_shape=None):
    """
    Determine and store the input and output shapes for each layer in a model by running it on a 'meta' device.
    This function creates a deep copy of the model, runs it on a 'meta' device to determine
    input and output shapes, and then stores these shapes as attributes (`input_shape` and `output_shape`)
    in the original model's layers.

    Args:
        model (torch.nn.Module): The model to analyze.
        input_shape (tuple, optional): The expected input shape to the model. If not provided,
                                       it must be stored as `input_shape` attribute in the model.
        get_output_shape (callable, optional): A function to determine the output shape
                                               for cases where `output` doesn't have a `shape` attribute.

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
                module.input_shape = input[0].shape
            else:
                module.input_shape = tuple(inp.shape for inp in input)

            if isinstance(output, torch.Tensor):
                module.output_shape = output.shape
            elif isinstance(output, (list, tuple)):
                module.output_shape = tuple(out.shape for out in output if isinstance(out, torch.Tensor))
            elif isinstance(output, dict):
                module.output_shape = {k: v.shape for k, v in output.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError(f"Unsupported output type: {type(output)}")

        # Register hooks to capture input/output shapes
        hook_handles = []
        for name, layer in model_on_meta.named_modules():
            if not hasattr(layer, 'output_shape'):
                handle = layer.register_forward_hook(fw_hook)
                hook_handles.append(handle)

        # Perform a forward pass to trigger the hooks
        model_on_meta(inp)

    # Copy the input/output shapes back to the original model's layers
    for name, layer in model.named_modules():
        layer_on_meta = get_module_by_name(model_on_meta, name)

        if name != '' and hasattr(layer_on_meta, 'output_shape'):
            layer.input_shape = layer_on_meta.input_shape
            layer.output_shape = layer_on_meta.output_shape

    return {name: (layer.input_shape, layer.output_shape) for name, layer in model.named_modules() if
            hasattr(layer, 'output_shape')}

def _get_layer_activations(layer, mode):
    """
    Internal function to retrieve the stored activations
    (either inputs or outputs) from a layer.

    Args:
        layer (torch.nn.Module): The layer from which to retrieve activations.
        mode (str): The mode specifying whether to return 'activations' (outputs) or 'inputs'.

    Returns:
        torch.Tensor: The stored activations (either inputs or outputs) of the layer.

    Raises:
        ValueError: If an unsupported mode is provided.
    """

    if mode == 'activations' or mode == 'outputs':
        return layer.stored_activations
    elif mode == 'inputs':
        return layer.stored_inputs
    else:
        raise ValueError("Mode must be 'activations' or 'inputs'")

def get_layer_activations_from_batch(model, layers, data, output_device, mode, model_device=None):
    with torch.no_grad():
        if model_device is None:
            model_device = get_device(model)
        model(data.to(model_device))

        activations_batch = {}

        for name, layer in layers.items():
            if name not in activations_batch.keys():
                activations_batch[name] = []

            activations_batch[name] = _get_layer_activations(layer, mode).to(output_device)
    return activations_batch

def get_layer_activations_from_dataset(model, layers, loader, output_device, mode, model_device=None):
    """
    Compute activations for specified layers across an entire dataset using the provided DataLoader.

    Args:
        model (torch.nn.Module): The model through which to pass the inputs.
        layers (dict): A dictionary where keys are layer names and values are layer modules
                       whose activations will be computed and stored.
        loader (torch.utils.data.DataLoader): DataLoader containing the dataset.
        output_device (torch.device): The device where the activations will be stored.
        model_device (torch.device, optional): The device on which the model and data reside.
                                               If None, it defaults to the model's device.
        mode (str): Specifies whether to retrieve 'activations' (outputs) or 'inputs'.

    Returns:
        dict: A dictionary with layer names as keys and a list of activations as values.
    """

    with torch.no_grad():

        if model_device is None:
            model_device = get_device(model)

        activations = {}

        for data, _ in loader:

            activations_batch = (
                get_layer_activations_from_batch(model, layers, data, output_device, mode, model_device))
            for layer_name in layers:
                if layer_name not in activations.keys():
                    activations[layer_name] = []

                activations[layer_name].append(activations_batch[layer_name])

        return activations

from time import time
def train_saes_in_batches(model, layers, saes, loader_model, sparsity_weight, learning_rate, num_epochs, device, mode):
    """
    Train multiple Sparse Autoencoders (SAEs) on the activations from different layers of a model using batches.

    Args:
        model (torch.nn.Module): The original model whose activations are used to train the SAEs.
        layers (dict): Dict of layers with names whose activations are used for training.
        saes (dict): Dictionary mapping layer names to corresponding SAE models.
        loader_model (torch.utils.data.DataLoader): DataLoader providing batches of input data for the original model.
        sparsity_weight (float): The weight for the sparsity penalty in the loss function.
        learning_rate (float): The learning rate for the optimizers.
        num_epochs (int): Number of epochs to train the SAEs.
        device (torch.device): The device to perform the training on.
        mode (str, optional): Mode specifying whether to train on 'activations' (outputs) or 'inputs'. Defaults to 'activations'.

    Returns:
        None
    """
    assert len(layers) == len(saes)

    optimizers = {name: optim.Adam(sae.parameters(), lr=learning_rate) for name, sae in saes.items()}

    for epoch in range(num_epochs):
        start_time = time()
        for batch_idx, (data, _) in enumerate(loader_model):
            activations = get_layer_activations_from_batch(model,
                                                           layers,
                                                           data.to(device),
                                                           output_device=device,
                                                           mode=mode
                                                           )

            for name, sae in saes.items():
                optimizer = optimizers[name]
                sae.train()

                optimizer.zero_grad()
                criterion = nn.MSELoss()

                # Forward pass for the SAE
                act = activations[name].flatten(1).to(device)
                encoded, decoded = sae(act)

                # Compute loss
                reconstruction_loss = criterion(decoded, act)
                sparsity_loss = torch.sum(torch.abs(encoded))
                loss = reconstruction_loss + sparsity_weight * sparsity_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print(f"Batch #{batch_idx}/{len(loader_model)}.")

        print(f'Epoch [{epoch + 1}/{num_epochs}] completed in {time() - start_time} seconds.')

    print('Training completed.')

def train_sae(sae, train_loader_sae, sparsity_weight, learning_rate, num_epochs, device, track_progress=True):
    """
    Train a Sparse Autoencoder (SAE) using the provided training data.

    Args:
        sae (torch.nn.Module): The Sparse Autoencoder model to be trained.
        train_loader_sae (torch.utils.data.DataLoader): DataLoader containing the training data.
        sparsity_weight (float): The weight for the sparsity penalty in the loss function.
        learning_rate (float): The learning rate for the optimizer.
        num_epochs (int): Number of epochs to train the model.
        device (torch.device): The device on which to perform the training.
        track_progress (bool, optional): Whether to print progress during training. Defaults to True.

    Returns:
        None
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        sae.train()
        total_loss, total_reconstruction_loss, total_weighted_sparcity_loss = 0, 0, 0

        for data in train_loader_sae:
            inputs = data[0].to(device)

            # Forward pass
            encoded, decoded = sae(inputs)

            # Compute loss
            reconstruction_loss = criterion(decoded, inputs)
            sparsity_loss = torch.sum(torch.abs(encoded))
            loss = reconstruction_loss + sparsity_weight * sparsity_loss

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_reconstruction_loss += reconstruction_loss.item()
            total_weighted_sparcity_loss += (sparsity_weight * sparsity_loss).item()
            total_loss += loss.item()

        if track_progress:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader_sae):.4f}')
            print(f'Reconstruction loss: {total_reconstruction_loss / len(train_loader_sae):.4f}')
            print(f'Weighted sparcity loss: {total_weighted_sparcity_loss / len(train_loader_sae):.4f}')

def create_sae(layer, device, *, mode=None, encoding_size=None, sae_class=SparseAutoencoder):
    """
    Create a Sparse Autoencoder (SAE) model tailored to the given layer.

    Args:
        layer (torch.nn.Module or int): The target layer or the size of the input.
        device (torch.device): Device the sae will be stored on.
        mode (str, optional): The mode specifying whether the SAE should be configured for the layer's inputs or outputs.
        encoding_size (int, optional): The size of the latent representation. Defaults to 4 * input size.
        sae_class (class, optional): The class of the SAE to instantiate. Defaults to SparseAutoencoder.

    Returns:
        torch.nn.Module: The initialized SAE model.

    Raises:
        ValueError: If `mode` is not specified when `layer` is a module.
    """

    if isinstance(layer, int):
        if encoding_size is None:
            encoding_size = layer * 3

        sae = sae_class(layer, encoding_size)

    else:

        input_size = torch.prod(torch.tensor(layer.input_shape[1:])).item()

        if encoding_size is None:
            encoding_size = input_size * 3

        if mode is None:
            raise ValueError("Mode('inputs'/'outputs') was not given.")

        if mode == 'inputs':
            sae = sae_class(input_size, encoding_size)
        elif mode == 'outputs' or mode == 'activations':
            sae = sae_class(input_size, encoding_size)
        else:
            raise ValueError('Unknown mode was given.')

    return sae.to(device)

def last_layer_sae_info(sae, activations, labels):
    """
    Evaluate the performance of a Sparse Autoencoder (SAE) on the last layer's activations.

    Args:
        sae (torch.nn.Module): The trained Sparse Autoencoder model.
        activations (torch.Tensor): Activations from the last layer of a model.
        labels (torch.Tensor): The corresponding labels for the activations.

    Returns:
        tuple: A tuple containing four lists:
            - both (list): Indices where both the original and reconstructed activations produce correct predictions.
            - only_model (list): Indices where only the original activations produce correct predictions.
            - only_rec (list): Indices where only the reconstructed activations produce correct predictions.
            - neither (list): Indices where neither the original nor reconstructed activations produce correct predictions.
    """

    both, only_model, only_rec, neither = ([] for i in range(4))

    with torch.no_grad():
        for i in range(len(activations)):
            model_activations, y = activations[i], labels[i]

            reconstructed_cnn_activations = sae(model_activations)[-1].detach().cpu()

            prediction = torch.argmax(model_activations).item()
            rec_prediction = torch.argmax(reconstructed_cnn_activations).item()

            if prediction == y:
                if rec_prediction == y:
                    both.append(i)
                else:
                    only_model.append(i)
            else:
                if rec_prediction == y:
                    only_rec.append(i)
                else:
                    neither.append(i)

    return both, only_model, only_rec, neither

def default_custom_remove(module, hook_handle, hook_name, is_pre_hook, message=0):
    if message == 0:
        message = f"{hook_name} hook was succesfully removed."

    if is_pre_hook:
        if hasattr(module, 'pre_hooks'):
            del module.pre_hooks[hook_name]
    else:
        if hasattr(module, 'hooks'):
            del module.hooks[hook_name]

    if message:
        print(message)

    hook_handle.remove()

def store_activations(module, hook_name, mode='activations', *, shape=None):
    """
    Create a hook to store activations (inputs or outputs) of a module.

    Args:
        module (torch.nn.Module): The module to attach the hook to.
        hook_name (str): Name of the hook.
        mode (str): Either 'activations' or 'inputs', specifying what to store.

    Returns:
        None
    """

    # Determine the appropriate attribute and message based on the mode
    if mode == 'activations' or mode == 'outputs':
        storage_attr = 'stored_activations'
        warning_message = "Output activations are already being stored."
        shape_name = 'output_shape'
    elif mode == 'inputs':
        storage_attr = 'stored_inputs'
        warning_message = "Input activations are already being stored."
        shape_name = 'input_shape'
    else:
        raise ValueError("Invalid mode. Use 'activations'/'outputs' or 'inputs'.")

    # Check if the module is already storing data
    if hasattr(module, storage_attr):
        if hasattr(module, 'hooks'):
            if hook_name not in module.hooks:
                print(f"Warning: The module has attribute {storage_attr} but {hook_name} is not in the hooks list of the module.")
            else:
                print(warning_message)
            print("New hook will not be created.")
            return
        else:
            raise ValueError(f"Error: Module has the attribute '{storage_attr}' but no hooks list.")

    if shape is None:
        if hasattr(module, shape_name):
            shape = getattr(module, shape_name)
        else:
            raise ValueError(f"No {shape_name} was provided and the module doesn't have the attribute .{shape_name}.")

    # Initialize storage for activations or inputs
    if mode == 'activations':
        module.stored_activations = torch.zeros(shape)
    elif mode == 'inputs':
        module.stored_inputs = torch.zeros(shape)

    # Define the hook function based on the mode
    def storing_hook(module, input, output):
        if mode == 'activations':
            module.stored_activations = output
        elif mode == 'inputs':
            if len(input) != 1:
                print("Warning: Non-standard input shape or something.")
            module.stored_inputs = input[0]

    # Register the forward hook
    hook_handle = module.register_forward_hook(storing_hook)

    # Define the custom remove function
    def custom_remove():
        if hasattr(module, storage_attr):
            delattr(module, storage_attr)
        if hasattr(module, 'hooks'):
            del module.hooks[hook_name]
            print(f"Storing {mode} hook was successfully removed.")
        hook_handle.remove()

    # Attach the custom remove function to the hook handle
    hook_handle.custom_remove = custom_remove

    # Ensure the module has a hooks dictionary and store the hook
    if not hasattr(module, 'hooks'):
        module.hooks = {}

    module.hooks[hook_name] = hook_handle

def custom_input_pre_hook(module, func, hook_name='custom_input'):
    def hook(module, input):
        input_to_substitute = func(input)
        if hasattr(input[0], 'shape'):
            assert input[0].shape == input_to_substitute.shape

        return (input_to_substitute,)

    hook_handle = module.register_forward_pre_hook(hook)

    # Register the forward hook
    if not hasattr(module, 'pre_hooks'):
        module.pre_hooks = {}

    module.pre_hooks[hook_name] = hook_handle
    # Attach the custom remove function to the hook handle
    hook_handle.custom_remove = lambda: default_custom_remove(module, hook_handle, hook_name, is_pre_hook=True)

def custom_input_pre_hook_old(module, input_to_substitute, hook_name='custom_input'):
    # deprecated
    def hook(module, input):
        if hasattr(input[0], 'shape'):
            assert input[0].shape == input_to_substitute.shape

        return (input_to_substitute,)

    hook_handle = module.register_forward_pre_hook(hook)

    # Register the forward hook
    if not hasattr(module, 'pre_hooks'):
        module.pre_hooks = {}

    module.pre_hooks[hook_name] = hook_handle
    # Attach the custom remove function to the hook handle
    hook_handle.custom_remove = lambda: default_custom_remove(module, hook_handle, hook_name, is_pre_hook=True)

def print_registered_hooks(model, model_name=None, *, old=False):
    """
    Print all registered forward and pre-hooks in the model.

    Args:
        model (torch.nn.Module): The model whose hooks are to be printed.
        model_name (str, optional): The name of the model. If not provided, the model's class name is used.
        old (bool, optional): Whether to use the old version of this function. Defaults to False.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    print(f"Registered hooks of {model_name}:")

    def print_hooks(hooks, hook_type, module_name):
        if hooks:
            for hook_name, hook in hooks.items():
                print(f"\tModule: {module_name}, {hook_type} name/ID: {hook_name}")

    if hasattr(model, 'hooks'):
        print_hooks(model.hooks, "Forward Hook", model_name)
    if hasattr(model, 'pre_hooks'):
        print_hooks(model.pre_hooks, "Pre-Hook", model_name)

    for name, module in model.named_children():
        module_name = f"{model_name}.{name}"
        
        if hasattr(module, 'pre_hooks') and not old:
            pre_hooks = module.pre_hooks
            if len(pre_hooks) != len(module._forward_pre_hooks):
                print("Warning: A discrepancy between pre_hooks dict and _forward_pre_hooks was detected.")
        else:
            pre_hooks = module._forward_pre_hooks

        if hasattr(module, 'hooks') and not old:
            hooks = module.hooks
            if len(hooks) != len(module._forward_hooks):
                print("Warning: A discrepancy between hooks dict and _forward_hooks was detected.")
        else:
            hooks = module._forward_hooks

        print_hooks(pre_hooks, "Pre-Hook", module_name)
        print_hooks(hooks, "Forward Hook", module_name)

def remove_all_forward_hooks(model):
    """
    Remove all forward hooks from the model.

    Args:
        model (torch.nn.Module): The model from which to remove all forward hooks.
    """

    for child in model.children():
        if child is not None:
            if hasattr(child, 'hooks'):
                child_hooks_copy = child.hooks.copy()

                for name, handle in child_hooks_copy.items():
                    if hasattr(handle, 'custom_remove'):
                        handle.custom_remove()
                    else:
                        handle.remove()

                del child.hooks

                if hasattr(child, "_forward_hooks"):
                    child._forward_hooks = OrderedDict()

            remove_all_forward_hooks(child)

def remove_all_forward_pre_hooks(model):
    """
    Remove all forward pre_hooks from the model.

    Args:
        model (torch.nn.Module): The model from which to remove all forward pre_hooks.
    """

    for child in model.children():
        if child is not None:
            if hasattr(child, 'pre_hooks'):
                child_pre_hooks_copy = child.pre_hooks.copy()

                for name, handle in child_pre_hooks_copy.items():
                    if hasattr(handle, 'custom_remove'):
                        handle.custom_remove()
                    else:
                        handle.remove()

                del child.pre_hooks

                if hasattr(child, "_forward_pre_hooks"):
                    child._forward_pre_hooks = OrderedDict()

            remove_all_forward_pre_hooks(child)

def optim_inputs_for_neurons_in_layer(model, layer, device, mode='activations', neuron_indices=None, init_inputs=None,
                                      alpha=20.0, image_sparsity=50.0, num_steps=1000):
    """
    Optimize input images to maximize the activation of specific neurons in a given layer.
    For correct functioning, the `layer` has to have the attribute stored_activations/stored_inputs (depending on the mode).
    The function save_layer_shapes can be used.

    Args:
        model (torch.nn.Module): The model containing the target layer.
        layer (torch.nn.Module): The layer for which neurons are to be optimized.
        device (torch.device): The device to run the optimization on.
        mode (str): Whether to optimize based on 'activations' (outputs) or 'inputs'.
        neuron_indices (torch.Tensor): Indices of the neurons to optimize for.
        init_inputs (str or torch.Tensor, optional): Initialization strategy or tensor for the inputs.
                                                     If 'random', inputs are initialized randomly.
                                                     If 'zeros', inputs are initialized to zero.
                                                     If a tensor is provided, it is used as the initial input.
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

    rotation_angles = [5, 10, 15]

    activations_to_optimize_for = _get_layer_activations(layer_on_device, mode)
    print(f"Initial activations: {_get_layer_activations(layer_on_device, mode)}")

    if max(neuron_indices) > activations_to_optimize_for.shape[-1]:
        print("Warning: neuron_indices out of range.")

    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        model_on_device(optimized_inputs)
        activations_to_optimize_for = _get_layer_activations(layer_on_device, mode)
        batch_of_activations = activations_to_optimize_for[:len(neuron_indices)]

        target_neurons_activations = torch.sum(batch_of_activations[torch.arange(len(neuron_indices)), neuron_indices])
        original_loss = torch.sum(batch_of_activations) - alpha * target_neurons_activations
        total_loss = original_loss + image_sparsity * torch.sum(optimized_inputs)

        for angle in rotation_angles:
            rotated_images = TF.rotate(optimized_inputs, angle)
            model_on_device(rotated_images)
            rotated_activations = _get_layer_activations(layer_on_device, mode)
            batch_of_rotated_activations = rotated_activations[:len(neuron_indices)]
            target_neurons_activations = torch.sum(
                batch_of_rotated_activations[torch.arange(len(neuron_indices)), neuron_indices])
            rotation_loss = torch.sum(batch_of_rotated_activations) - alpha * target_neurons_activations
            total_loss += rotation_loss + image_sparsity * torch.sum(rotated_images)

        total_loss.backward()
        optimizer.step()

        if (step + 1) % (num_steps // 10) == 0:
            print(f"step#{step + 1}")
            print(
                f"Original loss: {original_loss.item()}, Rotation loss: {rotation_loss.item()}, Sparsity loss: {image_sparsity * torch.sum(optimized_inputs).item()}")
            print(f"Activations: {_get_layer_activations(layer_on_device, mode)}")

    return optimized_inputs.detach()

def fgsm_attack(data, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # print(f"data_grad = {data_grad}")
    # print(f"sign_data_grad = {sign_data_grad}")
    # Create the perturbed image by adjusting each pixel of the input image
    delta = epsilon * sign_data_grad
    perturbed_data = data + delta
    # Adding clipping to maintain [0,1] range
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    # Return the perturbed image
    return perturbed_data

def fgsm_test(model, epsilon, loader, device, *, num_to_store=5, to_store_init_data=True):
    """
    Evaluate the model under adversarial attack using the FGSM method.

    Args:
        model (torch.nn.Module): The model to evaluate.
        epsilon (float): The magnitude of the adversarial perturbation.
        loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device to run the evaluation on.
        num_to_store (int): Number of examples to store for each epsilon.
        to_store_init_data (bool, optional): If True, stores the initial data before perturbation.

    Returns:
        float: The accuracy of the model under attack.
        list: A list of tuples containing the initial predictions, final predictions,
              and adversarial examples, optionally including the original data if `to_store_init_data` is True.
    """

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, targets in loader:

        # Set requires_grad attribute of tensor. Important for Attack
        data = data.to(device)
        targets = targets.to(device)
        data.requires_grad = True
        # Forward pass the data through the model
        output = model(data)

        init_preds = output.max(1)[1]  # get the index of the max

        if epsilon != 0:
            # Calculate the loss
            vectorized_targets = torch.eye(10).to(device)[targets]
            loss = F.mse_loss(output, vectorized_targets)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect ``datagrad``
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad).to(device)

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_preds = output.max(1)[1]  # get the index of the max log-probability
            # shape: (B,)

            with torch.no_grad():
                correct += torch.sum(final_preds == targets)

                if len(adv_examples) < num_to_store:
                    adv_ex_batch = perturbed_data.squeeze(0).cpu()  # shape: B, shape of data

                    mask = ((init_preds == targets) & (final_preds != targets)).cpu()

                    examples_to_add = num_to_store - len(adv_examples)

                    # Common elements to include in both cases
                    tuple_to_zip = [
                        init_preds[mask][:examples_to_add],
                        final_preds[mask][:examples_to_add],
                        adv_ex_batch[mask][:examples_to_add],
                    ]

                    # Add data only if `to_store_init_data` is False
                    if to_store_init_data:
                        tuple_to_zip.append(data[mask][:examples_to_add])

                    # Extend `adv_examples` with the appropriate elements
                    adv_examples.extend(zip(*tuple_to_zip))


        else:  # epsilon == 0
            with torch.no_grad():
                mask = (init_preds == targets).cpu()
                correct += torch.sum(mask)
                # Special case for saving 0 epsilon examples

                if len(adv_examples) < num_to_store:
                    data = data.squeeze(0).cpu()
                    examples_to_add = num_to_store - len(adv_examples)

                    # Common elements to include in both cases
                    tuple_to_zip = [
                        init_preds[mask][:examples_to_add],
                        init_preds[mask][:examples_to_add],
                        data[mask][:examples_to_add]
                    ]

                    if to_store_init_data:
                        tuple_to_zip.append(init_preds[mask][
                                            :examples_to_add])  # note: inefficient, done to have the same structure as in the eps != 0 case.

                    # Extend `adv_examples` with the appropriate elements
                    adv_examples.extend(zip(*tuple_to_zip))

    # Calculate final accuracy for this epsilon
    final_acc = correct / float(len(loader) * loader.batch_size)
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(loader) * loader.batch_size} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc.item(), adv_examples

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors

def visualize_ims(images, channel, titles=None, fig_title=None, cmap='coolwarm', norm=matplotlib.colors.CenteredNorm(), show_colorbar=None, figsize=None):
    """
    Visualize a batch of images along a specific channel.

    Args:
        images (torch.Tensor): A batch of images to visualize.
        channel (int): The channel of the images to visualize.
        titles (list of str, optional): Optional titles for the visualizations.
    """
    num_images = len(images)
    
    side_length = math.ceil(math.sqrt(num_images))
    
    if figsize is None:
        figsize = (side_length * 3, side_length * 3)
    
    fig, axes = plt.subplots(side_length, side_length, figsize=figsize)

    if num_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Flatten in case it's a 2D array

    for i in range(num_images):
        img = images[i][channel].cpu()
        ax = axes[i]
        im = ax.imshow(img.squeeze(), cmap=cmap, norm=norm)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i])

        # Use AxesDivider to add a colorbar next to the image
        if show_colorbar:
          divider = make_axes_locatable(ax)
          cax = divider.append_axes("right", size="5%", pad=0.05)
          plt.colorbar(im, cax=cax)

    # Hide any unused axes
    for j in range(num_images, len(axes)):
        axes[j].axis("off")

    if fig_title:
        plt.suptitle(fig_title, fontsize=16)

    plt.show()
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
