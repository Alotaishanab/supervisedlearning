#############################
# KERNEL PERCEPTRON Q7 with tqdm
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
def rbf_kernel(x1: cp.ndarray, x2: cp.ndarray, c: float) -> cp.ndarray:
    """
    Gaussian (RBF) kernel: exp(-c * ||x1 - x2||^2)
    x1: shape (N, D)
    x2: shape (M, D)
    c: positive float
    returns shape (N, M)
    """
    x1_sq = cp.sum(x1**2, axis=1).reshape(-1, 1)  # (N,1)
    x2_sq = cp.sum(x2**2, axis=1).reshape(1, -1)  # (1,M)
    sq_dists = x1_sq - 2*(x1 @ x2.T) + x2_sq
    return cp.exp(-c * sq_dists)


##########################################################
# 3. Multi-Class Kernel Perceptron (One-vs-All)
##########################################################
class MultiClassKernelPerceptron:
    """
    One-vs-All multi-class kernel perceptron with GPU acceleration.
    """
    def __init__(self, kernel_func, kernel_param, max_epochs=1):
        """
        kernel_func : rbf_kernel
        kernel_param: parameter 'c' for rbf
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
# 4. Q7 Procedure for Gaussian Kernel
##########################################################
def run_q7_rbf(
    X: np.ndarray, y: np.ndarray,
    c_values = [0.01, 0.1, 1, 5, 10],
    max_epochs=1,
    num_runs=20,
    train_ratio=0.8
):
    """
    Q7 for Gaussian kernel:
      - For each run:
         1) Randomly split into train (80%) + test (20%)
         2) For each c in c_values:
            - Train a Kernel Perceptron
            - Compute train error, test error
         3) Collect results
      - Finally, report mean±std across runs for each c.
    """
    train_errors_per_c = {c: [] for c in c_values}
    test_errors_per_c = {c: [] for c in c_values}

    for run_idx in tqdm(range(num_runs), desc="Q7 Runs"):
        # 1) Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_ratio=train_ratio, seed=run_idx
        )

        # 2) For each c, train & measure errors
        for c_val in c_values:
            model = MultiClassKernelPerceptron(
                kernel_func=rbf_kernel,
                kernel_param=c_val,
                max_epochs=max_epochs
            )
            model.fit(X_train, y_train)

            tr_err = model.score(X_train, y_train)
            te_err = model.score(X_test, y_test)

            train_errors_per_c[c_val].append(tr_err)
            test_errors_per_c[c_val].append(te_err)

    # Print summary
    print("=== Q7: Gaussian (RBF) Kernel ===")
    print("c\tMeanTrainErr ± Std\tMeanTestErr ± Std")
    for c_val in c_values:
        tr_arr = np.array(train_errors_per_c[c_val])
        te_arr = np.array(test_errors_per_c[c_val])
        print(f"{c_val}\t{tr_arr.mean():.4f}±{tr_arr.std():.4f}\t"
              f"{te_arr.mean():.4f}±{te_arr.std():.4f}")
    print()


##########################################################
# 5. Example Main to Solve Q7
##########################################################
def main_q7():
    # -----------------------------------
    #  LOAD DATA
    # -----------------------------------
    data_file = "zipcombo.dat"  # Ensure this file is uploaded to the current directory
    X, y = load_data_from_file(data_file)

    # -----------------------------------
    #  Q7: Gaussian Kernel
    # -----------------------------------
    run_q7_rbf(
        X, y,
        c_values=[0.01, 0.1, 1, 5, 10],
        max_epochs=1,
        num_runs=20,
        train_ratio=0.8
    )

# Run Q7
main_q7()

