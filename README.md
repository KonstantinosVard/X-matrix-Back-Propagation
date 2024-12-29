# Visualizing Classification Boundaries through Input Backpropagation

## Visualizing Classification Boundaries through Input Backpropagation

Welcome to this project, where I explore the boundaries of classification by employing backpropagation directly on the input matrix (X). Using a trained classifier, we delve into the regions of the feature space to understand where one class transitions into another. My goal is to shed light on how neural networks perceive data and transform it into decisions.

## What is this project about?

The core idea of this repository is, given a trained classifier, to backpropagate gradients through the input data to visualize and analyze the decision-making boundaries of a trained classifier. By doing so, we can transform images in ways that push them towards a specific classification target. This approach offers an intuitive and visual understanding of how models interpret inputs and what drives their predictions.

## How does it work?

I leverage gradient-based methods to explore the space around an input, focusing on two distinct strategies:

* Gradient Descent Across All Pixels (all_gradients):
  - Update the entire input matrix in every step to gradually morph an image towards a target class. Ideal for observing global shifts in feature representation.

* Selective Gradient Updates (one_gradient):
  - Adjust specific pixels with the highest gradient magnitudes to optimize the input for a target class. This method mimics fine-grained adjustments and highlights key areas influencing classification.

The methods are implemented with flexible stopping criteria, allowing the process to end either when a desired confidence level is reached or when the target class is successfully predicted.

## Files in This Repository

1) `adv_utils.py`:<br/>
   Contains helper utilities for training, evaluating, and visualizing predictions. Includes a lightweight neural network (KernelFitter) to experiment with gradient-based transformations.

2) `transform_image_to_target.py`:<br/>
        Implements the core functionality of backpropagating on the input matrix. Includes two functions, all_gradients and one_gradient, to perform global and localized updates respectively.

3) `adv_attack.ipynb`:<br/>
        A Jupyter notebook that ties everything together. Demonstrates the step-by-step process of transforming images to explore classification boundaries.

## Understanding the Value

Understanding how a model makes decisions is crucial for:

- Debugging and improving model performance.
- Ensuring model robustness against adversarial attacks.
- Enhancing interpretability for end-users and stakeholders.

This project bridges the gap between abstract model outputs and intuitive visual insights, making AI systems more transparent and trustworthy.
