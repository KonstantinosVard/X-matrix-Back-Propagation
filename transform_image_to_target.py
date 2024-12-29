import torch
from scipy.special import softmax

def all_gradients(model, image, target_label, steps=1000, learning_rate=0.1, class_pred_perc = None): ### add different stopage criteria
    """
    Transforms the input image to maximize the probability of the target label.
    Updates all pixels each time (step).

    Args:
        model: Trained PyTorch model.
        image: Input image tensor of shape [1, 1, 28, 28].
        target_label: The label to transform the image into.
        steps: Number of optimization steps.
        learning_rate: Step size for gradient updates.
        class_pred_perc: The probability that the new image belongs to the target class based on the classifier. 
                         If `None` the stopage happens when the predicted class is the target class.

    Returns:
        Transformed image tensor.
    """
    
    if class_pred_perc:
        def condition():
            probs = softmax(output.detach().numpy())[0][target_label]
            return (probs >= class_pred_perc)
    else:
        def condition():
            _, predicted = torch.max(output, 1)
            return (predicted.item() == target_label)
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Clone and set gradients for the input image
    image = image.clone().detach().requires_grad_(True)

    # Optimization loop
    for step in range(steps):
        model.zero_grad()

        # Forward pass
        output = model(image)
        
        # Directly maximize the logit of the target label
        loss = output[0, target_label]
        
        # Total Variation Regularization
        loss.backward()
        
        # Update the image
        with torch.no_grad():
            image += learning_rate * image.grad
            image.clamp_(-1, 1)  # Keep image pixel values in a valid range
            image.grad.zero_()

        # Print progress
        if step % 100 == 0:
            print(f"Step {step + 1}/{steps}, Logit for Target: {output[0, target_label].item():.4f}")
            
        # Check if stoppage condition is met:
        if condition():
            print(f"Target label {target_label} reached at step {step + 1}.")
            break

    return image



def one_gradient(model, image, target_label, steps=1000, learning_rate=0.1, max_grad_pixels=1, class_pred_perc=None):
    """
    Transforms the input image to maximize the probability of the target label.
    Updates one pixel (or `max_grad_pixels`) at a time (step).
    Args:
        model: Trained PyTorch model.
        image: Input image tensor of shape [1, 1, 28, 28].
        target_label: The label to transform the image into.
        steps: Number of optimization steps.
        learning_rate: Step size for gradient updates.
        early_stopping_threshold: Threshold to stop early if no progress is made.
        max_grad_pixels: Number of pixels to update based on gradients.
        class_pred_perc: The probability that the new image belongs to the target class based on the classifier. 
                         If `None` the stoppage happens when the predicted class is the target class.
    
    Returns:
        Transformed image tensor.
    """
    
    if class_pred_perc:
        def condition():
            probs = softmax(output.detach().numpy())[0][target_label]
            return (probs >= class_pred_perc)
    else:
        def condition():
            _, predicted = torch.max(output, 1)
            return (predicted.item() == target_label)
    
    model.eval()  # Ensure the model is in evaluation mode
    
    # Clone the image and ensure it requires gradients
    image = image.clone().detach().requires_grad_(True)
    
    prev_logit = None

    # Optimization loop
    for step in range(steps):
        model.zero_grad()

        # Forward pass
        output = model(image)
        
        # Extract the logit for the target label
        target_logit = output[0, target_label]

        # If it's the first step, store the initial logit
        if prev_logit is None:
            prev_logit = target_logit.item()

        # Compute the loss (maximize the logit for the target label)
        loss = target_logit

        # Compute gradients for the input image with respect to the loss
        loss.backward()

        # Get the gradients of the image
        gradients = image.grad.abs()

        # Get the top `max_grad_pixels` indices with the highest gradients
        flat_gradients = gradients.view(-1)  # Flatten the gradient tensor
        top_grad_indices = flat_gradients.topk(max_grad_pixels).indices  # Get the top indices
        # Convert the flat indices back to 2D pixel coordinates
        top_grad_indices = top_grad_indices.cpu().numpy()  # Convert to numpy for easier manipulation

        rows, cols = [], []
        for idx in top_grad_indices:
            row, col = divmod(idx, image.size(3))  # Use divmod to get row, col coordinates
            rows.append(row)
            cols.append(col)

        # Update pixels (not just top pixels) for more significant change
        for row in range(image.size(2)):  # Iterate over all rows
            for col in range(image.size(3)):  # Iterate over all columns
                # Update pixel values based on gradient
                image.data[0, 0, row, col] += learning_rate * image.grad[0, 0, row, col]

        # **Clamp the image after each update** to make sure no pixel value goes below 0 or above 1
        image.data = image.data.clamp(-1, 1)

        # Zero the gradients for the next iteration
        image.grad.zero_()

        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step + 1}/{steps}, Logit for Target: {target_logit.item():.4f}")

        # Check if the stoppage condition is met
        if condition():
            print(f"Target label {target_label} reached at step {step + 1}.")
            break

        # Update prev_logit for the next iteration
        prev_logit = target_logit.item()

    return image