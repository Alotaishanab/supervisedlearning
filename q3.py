import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

# Example Usage:
if __name__ == "__main__":
    import numpy as np

    def load_data(filepath):
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

    # Load the full dataset
    X, y = load_data('zipcombo.dat')

    # Initialize lists to store error rates for each degree
    degrees = list(range(1, 8))  # d = 1 to 7
    num_runs = 20
    train_errors = {d: [] for d in degrees}
    test_errors = {d: [] for d in degrees}

    for run in range(1, num_runs + 1):
        print(f"Run {run}/{num_runs}")
        # Split the data into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=run, stratify=y
        )

        for d in degrees:
            print(f"  Training with degree={d}")
            # Initialize the Kernel Perceptron with polynomial kernel of degree d
            perceptron = KernelPerceptronOVR(kernel='polynomial', degree=d, epochs=10)

            # Train the model
            perceptron.fit(X_train, y_train)

            # Evaluate on training data
            train_error = perceptron.evaluate(X_train, y_train)
            train_errors[d].append(train_error * 100)  # Convert to percentage

            # Evaluate on test data
            test_error = perceptron.evaluate(X_test, y_test)
            test_errors[d].append(test_error * 100)  # Convert to percentage

            print(f"    Training Error: {train_error * 100:.2f}%")
            print(f"    Test Error: {test_error * 100:.2f}%")
        print("\n")

    # Calculate mean and standard deviation for each degree
    import statistics

    mean_train_errors = {}
    std_train_errors = {}
    mean_test_errors = {}
    std_test_errors = {}

    for d in degrees:
        mean_train_errors[d] = statistics.mean(train_errors[d])
        std_train_errors[d] = statistics.stdev(train_errors[d])
        mean_test_errors[d] = statistics.mean(test_errors[d])
        std_test_errors[d] = statistics.stdev(test_errors[d])

    # Print the results
    print("Mean and Standard Deviation of Training and Testing Error Rates:")
    print("Degree\tTrain Error (%)\tTest Error (%)")
    for d in degrees:
        print(f"{d}\t{mean_train_errors[d]:.2f}±{std_train_errors[d]:.2f}\t\t{mean_test_errors[d]:.2f}±{std_test_errors[d]:.2f}")

    # Optionally, save the results to a file for LaTeX table
    import csv

    with open('q3_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Degree', 'Mean Train Error (%)', 'Std Train Error (%)', 'Mean Test Error (%)', 'Std Test Error (%)'])
        for d in degrees:
            writer.writerow([
                d,
                f"{mean_train_errors[d]:.2f}",
                f"{std_train_errors[d]:.2f}",
                f"{mean_test_errors[d]:.2f}",
                f"{std_test_errors[d]:.2f}"
            ])

    print("\nResults have been saved to 'q3_results.csv'.")
