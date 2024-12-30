import cupy as cp
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class KernelPerceptronOVR_GPU:
    def __init__(self, kernel='polynomial', degree=3, c=1.0, epochs=10):
        """
        Initializes the Kernel Perceptron (One-vs-Rest) for GPU computation using CuPy.

        Parameters:
        - kernel: Type of kernel ('polynomial' or 'gaussian').
        - degree: Degree of the polynomial kernel.
        - c: Parameter for the Gaussian kernel.
        - epochs: Number of training epochs.
        """
        self.kernel = kernel
        self.degree = degree
        self.c = c
        self.epochs = epochs
        # For each class, store a dict with 'alphas': [...], 'support_vectors': [...]
        self.classifiers = defaultdict(lambda: {'alphas': [], 'support_vectors': []})

    def polynomial_kernel(self, X_sv, x):
        """
        Computes the polynomial kernel between a batch of support vectors X_sv 
        and a single vector x, all on the GPU.
          - X_sv: shape (n_sv, n_features)
          - x: shape (n_features,)
        Returns: shape (n_sv,)
        """
        return (X_sv.dot(x)) ** self.degree

    def gaussian_kernel(self, X_sv, x):
        """
        Computes the Gaussian (RBF) kernel between a batch of support vectors X_sv 
        and a single vector x, all on the GPU.
          - X_sv: shape (n_sv, n_features)
          - x: shape (n_features,)
        Returns: shape (n_sv,)
        """
        diff = X_sv - x  # shape (n_sv, n_features)
        return cp.exp(-self.c * cp.linalg.norm(diff, axis=1) ** 2)

    def fit(self, X, y):
        """
        Trains the Kernel Perceptron using One-vs-Rest strategy on the GPU.

        Parameters:
        - X: Training data, CuPy array of shape (n_samples, n_features).
        - y: Training labels, CuPy array of shape (n_samples,).
        """
        classes = cp.unique(y)       # GPU array of class labels
        classes_cpu = classes.get()  # Move to CPU for iteration over classes

        n_samples = X.shape[0]
        total_updates = self.epochs * n_samples

        # Single tqdm progress bar for entire training
        with tqdm(total=total_updates, desc="Training KernelPerceptronOVR", ncols=80) as pbar:
            for epoch in range(self.epochs):
                for idx in range(n_samples):
                    x = X[idx]        # shape (n_features,)
                    label = y[idx]    # scalar
                    # For each class in OvR
                    for cls in classes_cpu:
                        binary_label = cp.int8(1 if label == cls else -1)

                        alphas_list = self.classifiers[cls]['alphas']
                        sv_list = self.classifiers[cls]['support_vectors']

                        if len(alphas_list) > 0:
                            alphas_cp = cp.array(alphas_list, dtype=cp.float32)
                            X_sv_cp = cp.stack(sv_list, axis=0)  # shape (n_sv, n_features)

                            if self.kernel == 'polynomial':
                                kernels = self.polynomial_kernel(X_sv_cp, x)
                            elif self.kernel == 'gaussian':
                                kernels = self.gaussian_kernel(X_sv_cp, x)
                            else:
                                raise ValueError("Unsupported kernel type.")

                            activation = alphas_cp.dot(kernels)
                        else:
                            activation = cp.float32(0.0)

                        if activation == 0:
                            prediction = cp.int8(1)
                        else:
                            prediction = cp.sign(activation).astype(cp.int8)

                        if prediction != binary_label:
                            self.classifiers[cls]['alphas'].append(binary_label)
                            self.classifiers[cls]['support_vectors'].append(x)

                    # Update progress bar once per sample
                    pbar.update(1)

    def predict(self, X):
        """
        Predicts the class labels for the given input data (GPU-based).

        Parameters:
        - X: Test data, CuPy array of shape (n_samples, n_features).

        Returns:
        - predictions: CuPy array of shape (n_samples,) with predicted class labels.
        """
        n_samples = X.shape[0]
        predictions = cp.zeros(n_samples, dtype=cp.int32)

        # We'll gather the classes from the dict keys (CPU side).
        classes_cpu = list(self.classifiers.keys())

        for i in range(n_samples):
            x = X[i]
            scores = {}
            for cls in classes_cpu:
                alphas_list = self.classifiers[cls]['alphas']
                sv_list = self.classifiers[cls]['support_vectors']

                if len(alphas_list) > 0:
                    alphas_cp = cp.array(alphas_list, dtype=cp.float32)
                    X_sv_cp = cp.stack(sv_list, axis=0)  # shape (n_sv, n_features)

                    if self.kernel == 'polynomial':
                        kernels = self.polynomial_kernel(X_sv_cp, x)
                    elif self.kernel == 'gaussian':
                        kernels = self.gaussian_kernel(X_sv_cp, x)
                    else:
                        raise ValueError("Unsupported kernel type.")
                    activation = alphas_cp.dot(kernels)
                else:
                    activation = cp.float32(0.0)

                scores[cls] = activation

            # pick class with highest activation
            best_cls = None
            best_score = -1e30
            for c_ in scores:
                if scores[c_] > best_score:
                    best_score = scores[c_]
                    best_cls = c_
            predictions[i] = best_cls

        return predictions

    def evaluate(self, X, y_true):
        """
        Evaluates the classifier on the given data (GPU-based).

        Parameters:
        - X: Data to evaluate, CuPy array of shape (n_samples, n_features).
        - y_true: True labels, CuPy array of shape (n_samples,).

        Returns:
        - error_rate: (float) proportion of misclassified samples.
        """
        y_pred = self.predict(X)
        misclassified = (y_pred != y_true).astype(cp.float32)
        error_rate = cp.mean(misclassified).item()  # item() to bring scalar to CPU
        return error_rate

# ========================
# Example usage (main code)
# ========================

if __name__ == "__main__":
    import numpy as np
    import cupy as cp
    from sklearn.model_selection import train_test_split

    def load_data(filepath):
        """
        Loads data from a .dat file and returns (X, y) on the GPU.
        
        Returns:
        - X: CuPy array of shape (n_samples, 256) if data is zipcombo-like.
        - y: CuPy array of shape (n_samples,).
        """
        # Load on CPU
        data_cpu = np.loadtxt(filepath)
        y_cpu = data_cpu[:, 0].astype(int)
        X_cpu = data_cpu[:, 1:]
        # Transfer to GPU
        X_gpu = cp.asarray(X_cpu, dtype=cp.float32)
        y_gpu = cp.asarray(y_cpu, dtype=cp.int32)
        return X_gpu, y_gpu

    # Load dataset
    X, y = load_data('zipcombo.dat')

    # Split data with scikit-learn (CPU)
    X_cpu = X.get()
    y_cpu = y.get()
    X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(
        X_cpu, y_cpu, test_size=0.2, random_state=42, stratify=y_cpu
    )
    # Convert back to GPU
    X_train = cp.asarray(X_train_cpu, dtype=cp.float32)
    y_train = cp.asarray(y_train_cpu, dtype=cp.int32)
    X_test = cp.asarray(X_test_cpu, dtype=cp.float32)
    y_test = cp.asarray(y_test_cpu, dtype=cp.int32)

    # Hyperparams
    kernel_types = ['polynomial']  # Or 'gaussian'
    epochs_list = [25]
    polynomial_degrees = [4]

    best_test_error = float('inf')
    best_train_error = None
    best_params = {}

    for kernel in kernel_types:
        if kernel == 'polynomial':
            param_values = polynomial_degrees
            param_name = 'degree'
        else:
            # If we do gaussian kernel, define e.g. gaussian_cs = [0.1, 1.0, 5.0]
            # param_values = gaussian_cs
            # param_name = 'c'
            raise NotImplementedError("Add code for Gaussian if desired.")

        for epochs in epochs_list:
            for param in param_values:
                print(f"Training with kernel={kernel}, {param_name}={param}, epochs={epochs}\n")

                perceptron = KernelPerceptronOVR_GPU(
                    kernel=kernel,
                    degree=param if kernel == 'polynomial' else 3,
                    c=param if kernel == 'gaussian' else 1.0,
                    epochs=epochs
                )

                # Fit model
                perceptron.fit(X_train, y_train)

                # Evaluate
                train_error = perceptron.evaluate(X_train, y_train)
                test_error = perceptron.evaluate(X_test, y_test)

                print(f"Training Error: {train_error * 100:.2f}%")
                print(f"Test Error:     {test_error * 100:.2f}%\n")

                # Track best
                if test_error < best_test_error:
                    best_test_error = test_error
                    best_train_error = train_error
                    best_params = {
                        'kernel': kernel,
                        param_name: param,
                        'epochs': epochs
                    }

    print("Best Model:")
    print(f"Kernel Type: {best_params['kernel']}")
    if best_params['kernel'] == 'polynomial':
        print(f"Degree: {best_params['degree']}")
    else:
        print(f"c: {best_params['c']}")
    print(f"Epochs: {best_params['epochs']}")
    print(f"Training Error: {best_train_error * 100:.2f}%")
    print(f"Test Error:     {best_test_error * 100:.2f}%")
