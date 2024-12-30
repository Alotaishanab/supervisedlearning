import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import statistics
import csv
from collections import defaultdict

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

# Example Usage:
if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import KFold
    from collections import defaultdict

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

    # Initialize variables to store results for Q4 and Q5
    degrees = list(range(1, 8))  # d = 1 to 7
    num_runs = 20
    selected_ds = []
    train_errors = []
    test_errors = []
    
    # Initialize confusion matrices for Q5
    confusion_matrices = []
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    # Initialize a list to hold confusion matrices for each run
    for run in range(1, num_runs + 1):
        print(f"Run {run}/{num_runs}")
        # Split the data into 80% training and 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=run, stratify=y
        )

        # 5-Fold Cross-Validation to select d*
        kf = KFold(n_splits=5, shuffle=True, random_state=run)
        cv_errors = {d: [] for d in degrees}

        for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
            X_cv_train, X_cv_val = X_train[train_index], X_train[val_index]
            y_cv_train, y_cv_val = y_train[train_index], y_train[val_index]

            for d in degrees:
                # Initialize the Kernel Perceptron with current degree
                perceptron = KernelPerceptronOVR(kernel='polynomial', degree=d, epochs=25)
                perceptron.fit(X_cv_train, y_cv_train)
                # Evaluate on validation set
                val_error = perceptron.evaluate(X_cv_val, y_cv_val) * 100  # Convert to percentage
                cv_errors[d].append(val_error)
        
        # Compute average validation error for each degree
        avg_cv_errors = {d: np.mean(cv_errors[d]) for d in degrees}
        # Select the degree with the lowest average validation error
        d_star = min(avg_cv_errors, key=avg_cv_errors.get)
        selected_ds.append(d_star)
        print(f"  Selected d* = {d_star} with Validation Error = {avg_cv_errors[d_star]:.2f}%")

        # Retrain on the full training set with d*
        final_perceptron = KernelPerceptronOVR(kernel='polynomial', degree=d_star, epochs=25)
        final_perceptron.fit(X_train, y_train)

        # Evaluate on training set
        final_train_error = final_perceptron.evaluate(X_train, y_train) * 100  # Convert to percentage
        train_errors.append(final_train_error)

        # Evaluate on test set
        final_test_error = final_perceptron.evaluate(X_test, y_test) * 100  # Convert to percentage
        test_errors.append(final_test_error)
        print(f"    Training Error: {final_train_error:.2f}%")
        print(f"    Test Error: {final_test_error:.2f}%\n")

        # Generate confusion matrix for this run
        y_pred = final_perceptron.predict(X_test)
        confusion = defaultdict(int)
        for true_label, pred_label in zip(y_test, y_pred):
            if true_label != pred_label:
                confusion[(true_label, pred_label)] += 1
        # Normalize confusion matrix by the number of true instances per class
        counts_per_class = defaultdict(int)
        for true_label in y_test:
            counts_per_class[true_label] += 1
        confusion_rate = defaultdict(float)
        for (a, b), count in confusion.items():
            confusion_rate[(a, b)] = count / counts_per_class[a] * 100  # Percentage
        confusion_matrices.append(confusion_rate)

    # Calculate mean and std for d*, train errors, and test errors
    mean_d_star = statistics.mean(selected_ds)
    std_d_star = statistics.stdev(selected_ds) if num_runs > 1 else 0.0
    mean_train_error = statistics.mean(train_errors)
    std_train_error = statistics.stdev(train_errors) if num_runs > 1 else 0.0
    mean_test_error = statistics.mean(test_errors)
    std_test_error = statistics.stdev(test_errors) if num_runs > 1 else 0.0

    print("Cross-Validation Results:")
    print(f"Mean d*: {mean_d_star:.2f} ± {std_d_star:.2f}")
    print(f"Mean Training Error: {mean_train_error:.2f}% ± {std_train_error:.2f}%")
    print(f"Mean Test Error: {mean_test_error:.2f}% ± {std_test_error:.2f}%\n")

    # Save Q4 results to CSV
    with open('q4_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Run', 'Selected d*', 'Training Error (%)', 'Test Error (%)'])
        for run in range(num_runs):
            writer.writerow([run + 1, selected_ds[run], train_errors[run], test_errors[run]])

    # Compute average confusion matrix
    average_confusion = defaultdict(list)
    for confusion in confusion_matrices:
        for (a, b), rate in confusion.items():
            average_confusion[(a, b)].append(rate)
    
    # Calculate mean and std for each confusion cell
    final_confusion = {}
    for a in unique_classes:
        for b in unique_classes:
            if a == b:
                continue  # Diagonal will be 0
            key = (a, b)
            rates = average_confusion.get(key, [0.0] * num_runs)
            mean_rate = statistics.mean(rates)
            std_rate = statistics.stdev(rates) if num_runs > 1 else 0.0
            final_confusion[key] = (mean_rate, std_rate)

    # Save confusion matrix to CSV
    with open('q5_confusion_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['True \\ Pred'] + [str(cls) for cls in unique_classes]
        writer.writerow(header)
        for a in unique_classes:
            row = [str(a)]
            for b in unique_classes:
                if a == b:
                    row.append('0.00±0.00')
                else:
                    rate, std = final_confusion.get((a, b), (0.0, 0.0))
                    row.append(f"{rate:.2f}±{std:.2f}")
            writer.writerow(row)

    print("Cross-Validation and Confusion Matrix results have been saved to 'q4_results.csv' and 'q5_confusion_matrix.csv' respectively.")
