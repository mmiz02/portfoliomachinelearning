# Perceptron Activities

This week I started learning about the Artificial Neural Network (ANN) and the use of perceptrons in machine learning models. During the lecturecast, I ran the recommended Python activities on Jupyter Notebook to reinforce my learning, including:

In each case, the perceptron uses a step activation function which outputs '1' if the weighted sum of the inputs is equal or larger than '1', and outputs '0' otherwise.<br>
- simple_perceptron.ipynb: Implemented a basic perceptron to understand the fundamental mechanism of linear classification. Basically, a weighted sum of inputs is calculated and a step activation function is used to output a '1' or '0'. <br>
- perceptron_AND_operator.ipynb: Applied a perceptron model to the logical AND operation. The weights were updated iteratively using a learning rate of '0.1' until the total error across all inputs reached zero. This allowed the perceptron to converge on the correct decision boundary separating the outputs '1' and '0'. After training, the perceptron was able to correctly classify all inputs for the AND operation. <br>
- multi-layer Perceptron.ipynb: Explored multi-layer perceptrons (MLPs) using the Sigmoid activation function. This allowed modeling non-linear relationships and illustrated the power of hidden layers in solving problems that single-layer perceptrons cannot. This allowed the model to learn more complex and non-linear relationships within the data. The sigmoid activation function transforms the weighted sum of inputs into a value between '0' and '1'. This makes the model suitable for classification tasks. <br>

This activity reinforces the importance of selecting an appropriate model architecture depending on the structure and complexity of the dataset.

### Reflection:<br>
These activities helped me develop a clear understanding of the applicability and challenges associated with different datasets in machine learning: <br>
- Linearly separable datasets can be effectively handled by a simple perceptron. <br>
- Non-linear datasets require multi-layer perceptrons or other more complex models to capture the underlying relationships. <br>
- The choice of activation function and network architecture can significantly affect learning and convergence. <br>
- Iterative weight adjustments demonstrate the importance of prroper learning rates, initialisation, and handling of potential vanishing gradient issues. <br>

Overall, this activity reinforced my practical understanding of perceptrons and MLPs, and highlighted the limitations of single-layer models, the importanc of datas structure, and nthe role of activation functions in extending model capability.
