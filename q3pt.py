#############################
# KERNEL PERCEPTRON Q3 with tqdm
#############################
import cupy as cp
import numpy as np
from tqdm import tqdm  # Import tqdm
from typing import Tuple

##########################################################
# 1. Utility Functions: Loading & Splitting Data
##########################################################
def load_data_from_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a data file where:
      - The first value of each line is the label (digit).
      - The next 256 values are the flattened 16x16 pixel intensities (range -1 to 1).

    Returns
    -------
    X : np.ndarray of shape (N, 256)
    y : np.ndarray of shape (N,)
    """
    data_list = []
    labels_list = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 257:
                # skip malformed lines
                continue
            label = int(float(values[0]))
            pixels = list(map(float, values[1:]))
            labels_list.append(label)
            data_list.append(pixels)
    X = np.array(data_list, dtype=np.float32)
    y = np.array(labels_list, dtype=int)
    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, train_ratio=0.8, seed=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle data, then split into training set and test set by 'train_ratio'.
    """
    if seed is not None:
        np.random.seed(seed)
    idx = np.random.permutation(len(X))
    split_point = int(train_ratio * len(X))
    X_train, X_test = X[idx[:split_point]], X[idx[split_point:]]
    y_train, y_test = y[idx[:split_point]], y[idx[split_point:]]
    return X_train, X_test, y_train, y_test


##########################################################
# 2. Kernel Functions
##########################################################
def polynomial_kernel(x1: cp.ndarray, x2: cp.ndarray, d: int) -> cp.ndarray:
    """
    Polynomial kernel: K_d(x1, x2) = (x1 dot x2)^d
    x1: shape (N, D)
    x2: shape (M, D)
    returns shape (N, M)
    """
    dot_product = x1 @ x2.T
    return cp.power(dot_product, d)


##########################################################
# 3. Multi-Class Kernel Perceptron (One-vs-All)
##########################################################
class MultiClassKernelPerceptron:
    """
    One-vs-All multi-class kernel perceptron with GPU acceleration.
    """
    def __init__(self, kernel_func, kernel_param, max_epochs=1):
        """
        kernel_func : polynomial_kernel
        kernel_param: degree 'd' for polynomial
        max_epochs  : number of epochs (passes) over training set
        """
        self.kernel_func = kernel_func
        self.kernel_param = kernel_param
        self.max_epochs = max_epochs
        self.classes_ = None
        self.alphas_ = {}
        self.X_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Move data to GPU
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        # Unique classes
        self.classes_ = cp.unique(y_gpu)
        self.X_train_ = X_gpu
        n_samples = X_gpu.shape[0]

        # For each class, define a binary classification
        for cls in self.classes_:
            alpha_k = cp.zeros(n_samples, dtype=cp.float32)
            
            for epoch in range(self.max_epochs):
                for i in range(n_samples):
                    y_binary = cp.float32(1.0 if y_gpu[i] == cls else -1.0)
                    
                    # Compute kernel between all training samples and the current sample
                    K_col_i = self.kernel_func(
                        self.X_train_, 
                        self.X_train_[i].reshape(1, -1),
                        self.kernel_param
                    ).squeeze()
                    
                    # Perceptron output
                    score = cp.sum(alpha_k * K_col_i)
                    pred_label = cp.float32(1.0 if score >= 0 else -1.0)
                    
                    # Update if there's a mistake
                    if pred_label != y_binary:
                        alpha_k[i] += y_binary
            
            # Store alpha for the current class
            self.alphas_[int(cls.get())] = alpha_k

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using a one-vs-all scheme: pick class with highest score_k(x).
        score_k(x) = sum_j alpha_k[j] * K(X_train[j], x).
        """
        X_gpu = cp.asarray(X)
        n_test = X_gpu.shape[0]
        # K_test_train shape => (n_test, n_train)
        K_test_train = self.kernel_func(X_gpu, self.X_train_, self.kernel_param)

        # compute scores for each class
        all_scores = []
        for cls in self.classes_:
            alpha_k = self.alphas_[int(cls.get())]
            # (n_test, n_train) dot (n_train,) => (n_test,)
            scores_k = K_test_train.dot(alpha_k)
            all_scores.append(scores_k.reshape(-1,1))

        # shape (n_test, n_classes)
        all_scores = cp.hstack(all_scores)
        # pick argmax => index
        preds_idx = cp.argmax(all_scores, axis=1)
        # classes_[preds_idx] => actual class
        return self.classes_[preds_idx].get()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the error rate on dataset (X,y).
        """
        preds = self.predict(X)
        return np.mean(preds != y)


##########################################################
# 4. Q3 Procedure for Polynomial Kernel
##########################################################
def run_q3_polynomial(
    X: np.ndarray, y: np.ndarray,
    degrees = [1,2,3,4,5,6,7],
    max_epochs=1,
    num_runs=20,
    train_ratio=0.8
):
    """
    Q3 for polynomial kernel:
      - For each run:
         1) Randomly split into train (80%) + test (20%)
         2) For each d in [1..7]:
            - Train a Kernel Perceptron
            - Compute train error, test error
         3) Collect results
      - Finally, report mean±std across runs for each d.
    """
    # Storage for errors across runs
    train_errors_per_d = {d: [] for d in degrees}
    test_errors_per_d = {d: [] for d in degrees}

    for run_idx in tqdm(range(num_runs), desc="Q3 Runs"):
        # 1) Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_ratio=train_ratio, seed=run_idx
        )

        # 2) For each d, train & measure errors
        for d in degrees:
            model = MultiClassKernelPerceptron(
                kernel_func=polynomial_kernel,
                kernel_param=d,
                max_epochs=max_epochs
            )
            model.fit(X_train, y_train)

            tr_err = model.score(X_train, y_train)
            te_err = model.score(X_test, y_test)

            train_errors_per_d[d].append(tr_err)
            test_errors_per_d[d].append(te_err)

    # Print summary
    print("=== Q3: Polynomial Kernel ===")
    print("d\tMeanTrainErr ± Std\tMeanTestErr ± Std")
    for d in degrees:
        tr_arr = np.array(train_errors_per_d[d])
        te_arr = np.array(test_errors_per_d[d])
        print(f"{d}\t{tr_arr.mean():.4f}±{tr_arr.std():.4f}\t"
              f"{te_arr.mean():.4f}±{te_arr.std():.4f}")
    print()


##########################################################
# 5. Example Main to Solve Q3
##########################################################
def main_q3():
    # -----------------------------------
    #  LOAD DATA
    # -----------------------------------
    data_file = "zipcombo.dat"  # Ensure this file is uploaded to the current directory
    X, y = load_data_from_file(data_file)

    # -----------------------------------
    #  Q3: Polynomial Kernel
    # -----------------------------------
    run_q3_polynomial(
        X, y,
        degrees=[1,2,3,4,5,6,7],
        max_epochs=1,
        num_runs=20,
        train_ratio=0.8
    )

# Run Q3
main_q3()
