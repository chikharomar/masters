from joblib import Parallel, delayed
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import math
from pennylane.kernels import target_alignment
import time

# Define a function to compute the Gram (kernel) matrix in parallel
def get_gram_para(x1, x2, kernel):
    # Initialize an empty list to store the rows of the Gram matrix
    gram_matrix = []
    
    # Iterate over each element in the first dataset
    for _x1 in x1:
        # Use parallel computation to apply the kernel function to _x1 and each element of x2
        # - Parallel(n_jobs=16): Specifies 16 parallel jobs
        # - delayed(kernel)(_x1, _x2): Delays the execution of the kernel function for each pair (_x1, _x2)
        results = Parallel(n_jobs=16)(delayed(kernel)(_x1, _x2) for _x2 in x2)
        
        # Convert the results from a list to a numpy array for easier handling
        results = np.array(results)
        
        # Append the computed row of the Gram matrix to the list
        gram_matrix.append(results)
    
    # Convert the list of rows into a complete Gram matrix (2D numpy array) and return it
    return np.array(gram_matrix)

# Define a function to compute the Gram (kernel) matrix
def get_gram(x1, x2, kernel):
    # Use a nested list comprehension to compute the Gram matrix:
    # - Outer loop iterates over each element _x1 in x1.
    # - Inner loop iterates over each element _x2 in x2 and computes kernel(_x1, _x2).
    # - For each _x1, the result is a list of kernel evaluations with all elements of x2.
    # - These lists form the rows of the Gram matrix.
    return np.array([[kernel(_x1, _x2) for _x2 in x2] for _x1 in x1])
    # The final result is converted into a numpy array for efficient numerical operations.

# Define a function to calculate the accuracy of predictions
def acc(results, target):
    # Compute the number of incorrect predictions:
    # - `results - target` computes the difference between predicted and true labels.
    # - `np.count_nonzero(results - target)` counts the number of non-zero differences (i.e., incorrect predictions).
    
    # Calculate the accuracy:
    # - `len(target)` gives the total number of predictions.
    # - The fraction of incorrect predictions is `np.count_nonzero(results - target) / len(target)`.
    # - Subtracting this fraction from 1 gives the accuracy.
    return 1 - np.count_nonzero(results - target) / len(target)

# Define a function to calculate the accuracy of a classifier
def accuracy(classifier, X, Y_target):
    # Use the classifier's predict method to obtain predictions for the input data X
    results = classifier.predict(X)
    
    # Calculate the number of incorrect predictions:
    # - `results - Y_target` computes the difference between the predicted labels (`results`)
    #   and the true labels (`Y_target`).
    # - `np.count_nonzero(results - Y_target)` counts the number of non-zero elements,
    #   which corresponds to the number of incorrect predictions.
    
    # Calculate the accuracy:
    # - `len(Y_target)` gives the total number of samples in the dataset.
    # - The fraction of incorrect predictions is `np.count_nonzero(results - Y_target) / len(Y_target)`.
    # - Subtracting this fraction from 1 gives the accuracy as a proportion of correct predictions.
    return 1 - np.count_nonzero(results - Y_target) / len(Y_target)


# Define a function to tune an SVM classifier for the best hyperparameter C
def SVM_tunner(X, y, kernel, step_size=0.25):
    # Split the dataset into training and validation sets
    # - `train_test_split` splits the input features `X` and labels `y` into training and validation datasets.
    # - `train_size=0.8` specifies that 80% of the data is used for training, and the remaining 20% for validation.
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)

    # Initialize variables to keep track of the best accuracy and corresponding hyperparameter C
    best_acc = 0  # Highest validation accuracy observed so far
    best_para = 0  # Value of C that resulted in the best accuracy

    # Loop over candidate values of the SVM regularization parameter C
    # - `np.arange(step_size, 1 + step_size, step_size)` generates a range of C values from `step_size` to 1 (inclusive),
    #   with increments of `step_size`.
    for C in np.arange(step_size, 1 + step_size, step_size):
        # Convert the current C value to a float
        C = float(C)

        # Train an SVM classifier with the current value of C and the provided kernel
        # - `SVC` is the Scikit-learn Support Vector Classifier.
        # - The kernel function is defined as a lambda function that calls `qml.kernels.kernel_matrix`.
        svm = SVC(C=C, kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, kernel)).fit(X_train, y_train)

        # Compute the accuracy of the model on the training set
        train_accuracy = accuracy(svm, X_train, y_train)

        # Compute the accuracy of the model on the validation set
        valid_accuracy = accuracy(svm, X_val, y_val)

        # Update the best accuracy and corresponding C value if the current validation accuracy is better
        if valid_accuracy > best_acc:
            best_acc = valid_accuracy  # Update best accuracy
            best_para = C  # Update the best hyperparameter value

    # Train the final SVM classifier using the best C value and the entire dataset
    svm_final = SVC(C=best_para, kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, kernel)).fit(X, y)

    # Return the final trained SVM and the best C value
    return svm_final, best_para
# Define a function to tune an RBF kernel SVM for the best hyperparameters C and gamma
def SVM_tunner_rbf(X, y, gamma_min=0.001, gamma_max=1, step_size=0.25):
    """
    Tunes an SVM with RBF kernel by finding the best hyperparameters C and gamma.

    Parameters:
        X (array-like): Input feature data.
        y (array-like): Target labels.
        gamma_min (float): Minimum value for the RBF kernel parameter gamma.
        gamma_max (float): Maximum value for the RBF kernel parameter gamma.
        step_size (float): Step size for iterating over the C parameter.

    Returns:
        svm_final (SVC): Trained SVM model with the best hyperparameters.
        best_para (float): Best C value.
        best_sigma (float): Best gamma value.
    """
    # Split the dataset into training and validation sets
    # - `train_test_split` divides `X` and `y` into training and validation datasets.
    # - `train_size=0.8` means 80% of the data is used for training, and 20% for validation.
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)

    # Initialize variables to keep track of the best accuracy and corresponding parameters
    best_acc = 0       # Best validation accuracy observed so far
    best_para = 0      # Best value of C (regularization parameter)
    best_sigma = gamma_min  # Best value of gamma (RBF kernel parameter)

    # Iterate over candidate values for the regularization parameter C
    # - `np.arange(step_size, 1 + step_size, step_size)` generates a range of C values between step_size and 1, inclusive.
    for C in np.arange(step_size, 1 + step_size, step_size):
        # Iterate over candidate values for the RBF kernel parameter gamma
        # - `np.arange(gamma_min, gamma_max, 0.1)` generates a range of gamma values between gamma_min and gamma_max, with increments of 0.1.
        for gamma in np.arange(gamma_min, gamma_max, 0.1):
            # Convert C and gamma to floats for consistency
            gamma = float(gamma)
            C = float(C)

            # Train an SVM classifier with the current values of C and gamma
            # - `kernel='rbf'` specifies the use of the RBF (Gaussian) kernel.
            svm = SVC(C=C, kernel='rbf', gamma=gamma).fit(X_train, y_train)

            # Compute the accuracy of the model on the validation set
            valid_accuracy = accuracy(svm, X_val, y_val)

            # Update the best parameters if the current model achieves a higher validation accuracy
            if valid_accuracy > best_acc:
                best_acc = valid_accuracy  # Update the best accuracy
                best_para = C             # Update the best C value
                best_sigma = gamma        # Update the best gamma value

    # Train the final SVM classifier using the best hyperparameters on the entire dataset
    svm_final = SVC(kernel='rbf', C=best_para, gamma=best_sigma).fit(X, y)

    # Return the final trained SVM and the best hyperparameters
    return svm_final, best_para, best_sigma


# Define a function to calculate the half-cut entanglement entropy of a quantum state
def entanglement(x):
    """
    Calculates the half-cut entanglement entropy for a quantum state vector.

    Parameters:
        x (array-like): The quantum state vector, assumed to be normalized and of length 2^n.

    Returns:
        float: The half-cut entanglement entropy.
    """

    # Determine the size of one subsystem after the half-cut
    # - The total state vector length is split evenly between two subsystems.
    # - `size` represents the number of states in the second subsystem.
    size = int(len(x) / 2)

    # Reshape the quantum state vector into a 2D matrix
    # - The reshaped state has dimensions 2 x `size` (representing two subsystems).
    reshaped_state = np.reshape(x, (2, size))

    # Perform Singular Value Decomposition (SVD) on the reshaped state
    # - `np.linalg.svd` returns three components: U, S, and V.
    # - We are interested in `S`, the singular values of the matrix.
    singular_values = np.linalg.svd(reshaped_state)[1]

    # Initialize a variable to store the entanglement entropy
    sum = 0

    # Compute the entanglement entropy
    # - For each singular value:
    #   - Calculate its corresponding eigenvalue (square of the singular value).
    #   - Add the term `eigen_value * log(eigen_value)` to the entropy sum.
    for singular_value in singular_values:
        if singular_value != 0:  # Ignore zero singular values to avoid log(0)
            eigen_value = singular_value**2  # Eigenvalue is the square of the singular value
            sum += eigen_value * math.log(eigen_value)

    # Return the negative of the computed sum, as entanglement entropy is -Σ(p * log(p))
    return -sum


# Define a function to generate random parameters for a quantum circuit
def random_params(num_wires, num_layers):
    """
    Generate random variational parameters for a quantum circuit.

    Parameters:
        num_wires (int): The number of wires (qubits) in the quantum circuit.
        num_layers (int): The number of layers in the quantum circuit.

    Returns:
        np.ndarray: A 2D array of shape (num_layers, num_wires) containing random 
                    variational parameters uniformly sampled between 0 and 2π. 
                    The array is created with gradient tracking enabled if supported.
    """
    # Use numpy's `np.random.uniform` to generate random values
    # - The range is [0, 2π], ensuring the parameters cover a full rotation in radians.
    # - The shape of the output is (num_layers, num_wires), where each layer has 
    #   one parameter for each wire (qubit).
    # - `requires_grad=True` enables gradient tracking for use in optimization-based 
    #   variational quantum algorithms.
    return np.random.uniform(0, 2 * np.pi, (num_layers, num_wires), requires_grad=True)

# Define a function to compute the kernel-target alignment between a kernel and labeled data
def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """
    Computes the kernel-target alignment (KTA) between a kernel and the label structure of a dataset.

    Parameters:
        X (array-like): Input data matrix of shape (n_samples, n_features).
        Y (array-like): Labels corresponding to the input data, of shape (n_samples,).
        kernel (callable): A function that computes the kernel between data points.
        assume_normalized_kernel (bool): Whether the kernel is already normalized.
        rescale_class_labels (bool): Whether to rescale the class labels to account for class imbalance.

    Returns:
        float: The kernel-target alignment value, quantifying the similarity between the kernel
               and the ideal label structure.
    """

    # Compute the kernel matrix using the provided kernel function
    # - `qml.kernels.square_kernel_matrix` calculates the square kernel matrix for the input data X.
    # - If `assume_normalized_kernel` is True, the kernel is assumed to be normalized already,
    #   simplifying the computation.
    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    # Optionally rescale the class labels to account for class imbalance
    if rescale_class_labels:
        # Count the number of positive and negative class samples
        nplus = np.count_nonzero(np.array(Y) == 1)  # Number of positive samples
        nminus = len(Y) - nplus                    # Number of negative samples

        # Rescale the labels:
        # - Positive labels (1) are divided by the number of positive samples (nplus).
        # - Negative labels (-1 or 0) are divided by the number of negative samples (nminus).
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        # If no rescaling is needed, use the labels as-is
        _Y = np.array(Y)

    # Compute the ideal kernel matrix based on the labels
    # - Outer product of the (possibly rescaled) labels with themselves.
    # - Represents the ideal target similarity structure.
    T = np.outer(_Y, _Y)

    # Compute the numerator of the kernel-target alignment
    # - The numerator is the inner product (Frobenius dot product) of the kernel matrix K and the ideal matrix T.
    inner_product = np.sum(K * T)

    # Compute the normalization term for the kernel-target alignment
    # - The normalization is the product of the Frobenius norms of K and T.
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))

    # Calculate the kernel-target alignment by normalizing the inner product
    inner_product = inner_product / norm

    # Return the kernel-target alignment value
    return inner_product



# Define a function to generate a dataset and label it using a quantum kernel and a specific entanglement architecture
def Generator(architecture, number_qubits, number_instances, number_layers):
    """
    Generates a dataset and labels it based on a quantum kernel defined by the specified entanglement architecture.

    Parameters:
        architecture (list of tuples): Specifies the connectivity of the qubits for entanglement.
        number_qubits (int): The number of qubits in the quantum circuit.
        number_instances (int): The number of data points (instances) to generate.
        number_layers (int): The number of layers in the quantum circuit ansatz.

    Returns:
        X (ndarray): The generated input dataset of shape (number_instances, number_qubits).
        y (list): Labels for the dataset, determined by the quantum kernel.
        init_params (ndarray): The randomly initialized parameters for the quantum circuit.
        architecture (list of tuples): The connectivity pattern used for entanglement.
        z (ndarray): A randomly chosen reference point in the feature space.
        average_entanglement (float): The average entanglement entropy of the generated quantum states.
    """
    # Generate random input data (angles for quantum rotations) and a random reference point z
    X = np.random.uniform(0, 2 * math.pi, (number_instances, number_qubits))
    z = np.random.uniform(0, 2 * math.pi, number_qubits)
    n_qubits = number_qubits  # Store the number of qubits

    # Define the quantum circuit layer
    def layer(x, params, wires, i0=0, inc=1):
        """
        A single layer of the quantum embedding ansatz.

        Parameters:
            x (ndarray): Input angles for the layer.
            params (ndarray): Trainable parameters for the layer.
            wires (list): List of qubit indices.
            i0 (int): Initial index for input angles (default: 0).
            inc (int): Increment step for cycling through inputs (default: 1).
        """
        for j, wire in enumerate(wires):
            qml.RX(x[j], wires=[wire])  # Apply rotation gate RX with input data
            qml.RX(params[j], wires=[wire])  # Apply rotation gate RX with trainable parameter
        qml.broadcast(unitary=qml.CNOT, pattern=architecture, wires=wires)  # Apply entangling CNOT gates

    # Define the complete ansatz (multi-layered circuit)
    def ansatz(x, params, wires):
        """
        Constructs the full quantum circuit using multiple layers of the ansatz.

        Parameters:
            x (ndarray): Input angles for the ansatz.
            params (ndarray): Trainable parameters for the ansatz.
            wires (list): List of qubit indices.
        """
        for j, layer_params in enumerate(params):
            layer(x, layer_params, wires, i0=j * len(wires))

    # Adjoint (conjugate transpose) of the ansatz for specific tasks (not used in this function)
    adjoint_ansatz = qml.adjoint(ansatz)

    # Initialize random parameters for the ansatz
    init_params = random_params(num_wires=n_qubits, num_layers=number_layers)

    # Define a quantum device
    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)  # Simulation device with specified qubits
    wires_dev = dev.wires.tolist()

    # Define a QNode to compute the quantum state for a single data point
    @qml.qnode(dev)
    def single_feature_space(x, params):
        ansatz(x, params, wires=wires_dev)
        return qml.state()  # Return the quantum state

    # Compute the quantum feature space for the entire dataset
    def total_feature_space(X, params):
        results = [single_feature_space(_x, params) for _x in X]
        return results

    # Define a kernel function to compute inner products in the quantum feature space
    def inner(x, y):
        return np.absolute(np.vdot(x, y))**2  # Compute the squared inner product

    # Compute the feature space for the dataset and the reference point
    X_space = total_feature_space(X, init_params)
    z_feature = single_feature_space(z, init_params)

    # Compute the entanglement entropy for each quantum state
    entang = Parallel(n_jobs=8)(delayed(entanglement)(x_feature) for x_feature in X_space)
    average_entanglement = np.mean(entang)  # Compute the average entanglement entropy

    # Compute the inner products between the feature states and the reference point
    inner_products = Parallel(n_jobs=8)(delayed(inner)(element, z_feature) for element in X_space)

    # Compute the median of the inner products as the decision threshold
    gamma = np.median(inner_products)

    # Label the data based on the inner product threshold
    y = [-1 if ip_i < gamma else 1 for ip_i in inner_products]

    # Return the dataset, labels, initial parameters, architecture, reference point, and average entanglement
    return X, y, init_params, architecture, z, average_entanglement

# Define a function to generate a dataset using the Generator function and save it to a file
def Dataset_Saver(architecture, number_qubits, number_instances, number_layers):
    """
    Generates a dataset using the specified quantum architecture and saves it as a pickle file.

    Parameters:
        architecture (list of tuples): Specifies the connectivity of the qubits for entanglement.
        number_qubits (int): The number of qubits in the quantum circuit.
        number_instances (int): The number of data points (instances) to generate.
        number_layers (int): The number of layers in the quantum circuit ansatz.

    Returns:
        None
    """

    # Generate the dataset and associated parameters using the Generator function
    # - X: The generated dataset (features).
    # - y: Labels for the dataset.
    # - generation_param: Randomly initialized parameters for the quantum circuit.
    # - architecture: The entanglement architecture used for the circuit.
    # - z: A randomly chosen reference point in the quantum feature space.
    # - average_entanglement: The average entanglement entropy of the generated quantum states.
    X, y, generation_param, architecture, z, average_entanglement = Generator(
        architecture=architecture,
        number_qubits=number_qubits,
        number_instances=number_instances,
        number_layers=number_layers,
    )

    # Construct a filename for saving the dataset
    # - The filename includes the architecture, number of layers, and number of qubits for easy identification.
    file_name = architecture + "_" + str(number_layers) + "Layers_" + str(number_qubits) + "Qubits.pickle"

    # Save the dataset and associated parameters to a pickle file
    with open("Datasets_lightning/" + file_name, 'wb') as output:
        # Save a list containing the dataset and associated parameters:
        # - X: Features of the dataset.
        # - y: Labels for the dataset.
        # - generation_param: Initialization parameters for the quantum circuit.
        # - z: Reference point in the feature space.
        # - average_entanglement: Average entanglement entropy of the states.
        pickle.dump([X, y, generation_param, z, average_entanglement], output)

        # Print a confirmation message to indicate successful completion
        print("Done with : " + file_name)
# Define a function to load a dataset from a specified file
def Load_Dataset(architecture, number_qubits, number_layer):
    """
    Loads a dataset generated by the Dataset_Saver function from a pickle file.

    Parameters:
        architecture (str): The name of the quantum circuit architecture (used in the filename).
        number_qubits (int): The number of qubits in the quantum circuit (used in the filename).
        number_layer (int): The number of layers in the quantum circuit (used in the filename).

    Returns:
        list: The loaded dataset and associated parameters, including:
              - X: The dataset features.
              - y: The dataset labels.
              - generation_param: The initial quantum circuit parameters.
              - z: The reference point in the quantum feature space.
              - average_entanglement: The average entanglement entropy of the quantum states.
    """

    # Construct the full file path based on the given parameters
    # - The file path includes the directory "Datasets_lightning" and the filename, which is constructed
    #   using the architecture name, number of layers, and number of qubits.
    file_path = (
        "/home/mila/o/omar.chikhar/masters_main/Datasets_lightning/"
        + architecture
        + "_"
        + str(number_layer)
        + "Layers_"
        + str(number_qubits)
        + "Qubits.pickle"
    )

    # Open the pickle file for reading in binary mode
    file = open(file_path, 'rb')

    # Load the dataset and associated parameters from the pickle file
    # - The returned object is a list containing:
    #   - X: The dataset features.
    #   - y: The dataset labels.
    #   - generation_param: Initialization parameters for the quantum circuit.
    #   - z: Reference point in the quantum feature space.
    #   - average_entanglement: Average entanglement entropy of the states.
    return pickle.load(file)



# Define a function to train a quantum kernel using a specific dataset and architecture
def Train_Quantum_kernel_function(
    number_layers,
    number_qubits,
    target_architecture,
    target_number_of_layers,
    model_architecture,
    optimizer=qml.SPSAOptimizer,
    alpha_0=0.7,
    step_size=0.2
):
    """
    Trains a quantum kernel on a given dataset with a specified architecture.

    Parameters:
        number_layers (int): The number of layers in the quantum kernel ansatz.
        number_qubits (int): The number of qubits in the quantum circuit.
        target_architecture (list of tuples): The target architecture used to generate the dataset.
        target_number_of_layers (int): The number of layers in the dataset generation ansatz.
        model_architecture (list of tuples): The architecture used to define the kernel for training.
        optimizer (callable): The optimizer to use for training (default: qml.SPSAOptimizer).
        alpha_0 (float): Initial learning rate for the optimizer (default: 0.7).
        step_size (float): Step size for gradient-based optimization (default: 0.2).

    Returns:
        None
    """

    # Load the dataset generated with the target architecture
    X, y, generation_params, _, _ = Load_Dataset(
        architecture=target_architecture,
        number_layer=target_number_of_layers,
        number_qubits=number_qubits
    )
    n_qubits = number_qubits  # Number of qubits

    # Define a single layer of the quantum ansatz
    def layer(x, params, wires, i0=0, inc=1):
        for j, wire in enumerate(wires):
            qml.RX(x[j], wires=[wire])  # Rotation gate with input features
            qml.RX(params[j], wires=[wire])  # Rotation gate with trainable parameters
        # Apply entangling CNOT gates based on the model architecture
        qml.broadcast(unitary=qml.CNOT, pattern=model_architecture, wires=wires)

    # Define the full quantum ansatz with multiple layers
    def ansatz(x, params, wires):
        for j, layer_params in enumerate(params):
            layer(x, layer_params, wires, i0=j * len(wires))

    # Define the adjoint (conjugate transpose) of the ansatz
    adjoint_ansatz = qml.adjoint(ansatz)

    # Compute the inner product in the quantum feature space
    def inner(x, y):
        return np.absolute(np.vdot(x, y))**2

    # Define the quantum device for state preparation
    dev = qml.device("lightning.qubit", wires=n_qubits, shots=None)
    wires_dev = dev.wires.tolist()

    # Define a QNode for computing the quantum state of a single input
    @qml.qnode(dev)
    def single_feature_space(x, params):
        ansatz(x, params, wires=wires_dev)
        return qml.state()

    # Compute the quantum feature space for a dataset
    def total_feature_space(X, params):
        return [single_feature_space(_x, params) for _x in X]

    # Define a QNode for the kernel circuit
    dev = qml.device("default.qubit", wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2, params):
        ansatz(x1, params, wires=wires_dev)
        adjoint_ansatz(x2, params, wires=wires_dev)
        return qml.probs(wires=wires_dev)

    # Define the kernel function
    def kernel(x1, x2, params):
        return kernel_circuit(x1, x2, params)[0]

    # Define a helper function for target alignment
    def target_alignment(X, Y, kernel, assume_normalized_kernel=False, rescale_class_labels=False):
        K = qml.kernels.square_kernel_matrix(X, kernel, assume_normalized_kernel=True)
        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(Y) == 1)
            nminus = len(Y) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
        else:
            _Y = np.array(Y)
        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        return inner_product / norm

    # Initialize the parameters for the quantum kernel
    init_params = random_params(num_wires=n_qubits, num_layers=number_layers)
    init_kernel = lambda x1, x2: kernel(x1, x2, init_params)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1)
    X_train_small = X_train[:50]
    y_train_small = y_train[:50]

    # Compute the initial kernel target alignment (KTA)
    kta_init = qml.kernels.target_alignment(X_train_small, y_train_small, init_kernel, assume_normalized_kernel=True)
    print("Initial kernel alignment:", kta_init)

    # Optimize the kernel parameters using the target alignment as the objective
    params = init_params
    opt = qml.GradientDescentOptimizer(0.2)
    for i in range(100):
        # Define the cost function as the negative target alignment
        cost = lambda _params: -target_alignment(X_train_small, y_train_small, lambda x1, x2: kernel(x1, x2, _params), assume_normalized_kernel=True)
        params = opt.step(cost, params)
        current_alignment = target_alignment(X_train_small, y_train_small, lambda x1, x2: kernel(x1, x2, params), assume_normalized_kernel=True)
        print(f"Step {i} - Alignment = {current_alignment:.7f}")

        # Train and evaluate an SVM using the updated kernel
        kernel_function = lambda x1, x2: kernel(x1, x2, params)
        trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, kernel_function)
        svm_trained = SVC(kernel=trained_kernel_matrix).fit(X_train, y_train)

        # Compute the training and testing accuracy
        train_accuracy = accuracy(svm_trained, X_train, y_train)
        test_accuracy = accuracy(svm_trained, X_test, y_test)
        print(f"Iteration {i}: Train Accuracy = {train_accuracy:.3f}, Test Accuracy = {test_accuracy:.3f}")

