import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
import statistics
import matplotlib.pyplot as plt

class KernelPerceptronOVR:
    def __init__(self, kernel='polynomial', degree=3, c=1.0, epochs=10):
        """
        Initializes the Kernel Perceptron with One-vs-Rest strategy.

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
        self.classifiers = defaultdict(lambda: {'alphas': [], 'support_vectors': []})

    def polynomial_kernel(self, x, y):
        """Computes the polynomial kernel between two vectors."""
        return (np.dot(x, y)) ** self.degree

    def gaussian_kernel(self, x, y):
        """Computes the Gaussian (RBF) kernel between two vectors."""
        return np.exp(-self.c * np.linalg.norm(x - y) ** 2)

    def compute_kernel(self, x, y):
        """Selects and computes the specified kernel function."""
        if self.kernel == 'polynomial':
            return self.polynomial_kernel(x, y)
        elif self.kernel == 'gaussian':
            return self.gaussian_kernel(x, y)
        else:
            raise ValueError("Unsupported kernel type. Choose 'polynomial' or 'gaussian'.")

    def fit(self, X, y):
        """
        Trains the Kernel Perceptron using One-vs-Rest strategy.

        Parameters:
        - X: Training data, numpy array of shape (n_samples, n_features).
        - y: Training labels, numpy array of shape (n_samples,).
        """
        classes = np.unique(y)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            for idx, x in enumerate(X):
                label = y[idx]
                for cls in classes:
                    # Binary label for current classifier: +1 for class cls, -1 otherwise
                    binary_label = 1 if label == cls else -1
                    alphas = self.classifiers[cls]['alphas']
                    support_vectors = self.classifiers[cls]['support_vectors']

                    # Compute activation using vectorized operations
                    if len(alphas) > 0:
                        support_vectors_np = np.array(support_vectors)
                        alphas_np = np.array(alphas)
                        if self.kernel == 'polynomial':
                            kernels = np.dot(support_vectors_np, x) ** self.degree
                        elif self.kernel == 'gaussian':
                            # Efficient computation using broadcasting
                            diff = support_vectors_np - x
                            kernels = np.exp(-self.c * np.linalg.norm(diff, axis=1) ** 2)
                        activation = np.dot(alphas_np, kernels)
                    else:
                        activation = 0.0

                    # Prediction
                    prediction = np.sign(activation) if activation != 0 else 1

                    # Update rule
                    if prediction != binary_label:
                        self.classifiers[cls]['alphas'].append(binary_label)
                        self.classifiers[cls]['support_vectors'].append(x)
            print("Epoch completed.\n")

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Parameters:
        - X: Test data, numpy array of shape (n_samples, n_features).

        Returns:
        - predictions: Predicted class labels, numpy array of shape (n_samples,).
        """
        predictions = []
        classes = list(self.classifiers.keys())
        for idx, x in enumerate(X):
            scores = {}
            for cls in classes:
                alphas = np.array(self.classifiers[cls]['alphas'])
                support_vectors = np.array(self.classifiers[cls]['support_vectors'])
                if len(alphas) > 0:
                    if self.kernel == 'polynomial':
                        kernels = np.dot(support_vectors, x) ** self.degree
                    elif self.kernel == 'gaussian':
                        diff = support_vectors - x
                        kernels = np.exp(-self.c * np.linalg.norm(diff, axis=1) ** 2)
                    activation = np.dot(alphas, kernels)
                else:
                    activation = 0.0
                scores[cls] = activation
            # Assign the class with the highest activation score
            predicted_class = max(scores, key=scores.get)
            predictions.append(predicted_class)
        return np.array(predictions)

    def evaluate(self, X, y_true):
        """
        Evaluates the classifier on the given data.

        Parameters:
        - X: Data to evaluate, numpy array of shape (n_samples, n_features).
        - y_true: True labels, numpy array of shape (n_samples,).

        Returns:
        - error_rate: Proportion of misclassified samples.
        """
        y_pred = self.predict(X)
        error_rate = np.mean(y_pred != y_true)
        return error_rate

def load_data(filepath="zipcombo.dat"):
    """
    Loads data from a .dat file.

    Parameters:
    - filepath: Path to the .dat file.

    Returns:
    - X: Feature matrix, numpy array of shape (n_samples, 256).
    - y: Labels, numpy array of shape (n_samples,).
    """
    data = np.loadtxt(filepath)
    y = data[:, 0].astype(int)
    X = data[:, 1:]
    return X, y

def run_q6_demo(X, y, degrees=[1,2,3,4,5,6,7], num_runs=20, random_seed=42):
    """
    Performs Q6: Identifies the five hardest-to-predict samples based on misclassifications across runs.

    Parameters:
    - X: Feature matrix, numpy array of shape (n_samples, n_features).
    - y: Labels, numpy array of shape (n_samples,).
    - degrees: List of polynomial degrees to consider.
    - num_runs: Number of independent runs.
    - random_seed: Seed for reproducibility.

    Returns:
    - top5_indices: List of indices corresponding to the five hardest samples.
    - misCount: Array indicating how many times each sample was misclassified.
    """
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    misCount = np.zeros(n_samples, dtype=int)

    for run in range(1, num_runs + 1):
        print(f"Run {run}/{num_runs}")
        # Create an array of indices
        indices = np.arange(n_samples)
        # Split the data and the indices into training and testing
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, indices, test_size=0.2, random_state=run, stratify=y
        )

        # 5-Fold Cross-Validation to select d*
        kf = KFold(n_splits=5, shuffle=True, random_state=run)
        cv_errors = {d: [] for d in degrees}

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_cv_train, X_cv_val = X_train[train_index], X_train[val_index]
            y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]

            for d in degrees:
                # Initialize the Kernel Perceptron with current degree
                perceptron = KernelPerceptronOVR(kernel='polynomial', degree=d, epochs=10)
                perceptron.fit(X_cv_train, y_cv_train)
                # Evaluate on validation set
                val_error = perceptron.evaluate(X_cv_val, y_cv_val) * 100  # Convert to percentage
                cv_errors[d].append(val_error)

        # Compute average validation error for each degree
        avg_cv_errors = {d: np.mean(cv_errors[d]) for d in degrees}
        # Select the degree with the lowest average validation error
        d_star = min(avg_cv_errors, key=avg_cv_errors.get)
        print(f"  Selected d* = {d_star} with Validation Error = {avg_cv_errors[d_star]:.2f}%")

        # Retrain on the full training set with d*
        final_perceptron = KernelPerceptronOVR(kernel='polynomial', degree=d_star, epochs=10)
        final_perceptron.fit(X_train, y_train)

        # Evaluate on test set
        final_test_error = final_perceptron.evaluate(X_test, y_test) * 100  # Convert to percentage
        print(f"    Test Error: {final_test_error:.2f}%")

        # Generate confusion matrix for this run
        y_pred = final_perceptron.predict(X_test)
        for i, idx in enumerate(test_idx):
            if y_pred[i] != y_test[i]:
                misCount[idx] += 1
        print()

    # Identify the top 5 hardest samples
    top5_indices = misCount.argsort()[-5:][::-1]
    return top5_indices, misCount

def visualize_hardest_samples(X, y, top5_indices):
    """
    Visualizes the five hardest-to-predict samples.

    Parameters:
    - X: Feature matrix, numpy array of shape (n_samples, 256).
    - y: Labels, numpy array of shape (n_samples,).
    - top5_indices: List of indices corresponding to the five hardest samples.
    """
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(top5_indices):
        img = X[idx].reshape(16, 16)
        plt.subplot(1, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {y[idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # Load the dataset
    X, y = load_data("zipcombo.dat")
    print("Data loaded. Shape of X:", X.shape, "Shape of y:", y.shape)

    # Run Q6 to find the top 5 hardest samples
    top5_indices, misCount = run_q6_demo(X, y, num_runs=20)

    print("The 5 hardest samples are indices:", top5_indices)
    for i, idx in enumerate(top5_indices, 1):
        print(f"\nSample {i}:")
        print(f"Index: {idx}, Misclassified {misCount[idx]} times")
        print(f"True Label: {y[idx]}")
        # Reshape the sample as 16x16 for visualization
        img = X[idx].reshape(16, 16)
        plt.imshow(img, cmap='gray')
        plt.title(f"Sample {i} - True Label: {y[idx]}")
        plt.axis('off')
        plt.show()
