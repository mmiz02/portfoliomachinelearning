# Perceptron Activities

This week I started learning about the Artificial Neural Network (ANN) and the use of perceptrons in machine learning models. During the lecturecast, I ran the recommended Python activities on Jupyter Notebook to reinforce my learning, including:

In each case, the perceptron uses a step activation function which outputs '1' if the weighted sum of the inputs is equal or larger than 1, and outputs '0' otherwise.
- simple_perceptron.ipynb: Implemented a basic perceptron to understand the fundamental mechanism of linear classification. Basically, a weighted sum of inputs is calculated and a step activation function is used to output a '1' or '0'.
- perceptron_AND_operator.ipynb: Applied a perceptron model to the logical AND operation. The weights were updated iteratively using a learning rate of 0.1.














perceptron_AND_operator.ipynb – Applied a perceptron to model the logical AND operation. This activity demonstrated that simple linear decision boundaries are sufficient for linearly separable problems. It also illustrated the limitations of a single-layer perceptron when attempting non-linear problems, such as XOR.

multi-layer Perceptron.ipynb (Sigmoid Function) – Explored multi-layer perceptrons (MLPs) using the Sigmoid activation function. This allowed modeling non-linear relationships and illustrated the power of hidden layers in solving problems that single-layer perceptrons cannot. I observed how forward propagation, backpropagation, and weight updates worked together to minimize error.

Reflection

These activities helped me develop a clear understanding of the applicability and challenges associated with different datasets in machine learning:

Linearly separable datasets can be effectively handled by a simple perceptron.

Non-linear datasets require multi-layer perceptrons or other more complex models to capture the underlying relationships.

The choice of activation function (e.g., Sigmoid) and network architecture can significantly affect learning and convergence.

Iterative weight adjustments demonstrate the importance of proper learning rates, initialization, and handling of potential vanishing gradient issues.

Overall, these exercises reinforced my practical understanding of perceptrons and MLPs, and highlighted the limitations of single-layer models, the importance of data structure, and the role of activation functions in extending model capability.
