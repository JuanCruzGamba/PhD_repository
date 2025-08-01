# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# A class to perform Decision Tree classification on biomarker data, including cross-validation, metric calculation, and visualization.
class BiomarkerDecisionTree:
   
    def __init__(self, df_main: pd.DataFrame, target_column='survival_day_25'):
        """
        Initializes the BiomarkerDecisionTree object with a pre-processed DataFrame.

        Args:
            df_main (pd.DataFrame): The main DataFrame that is already loaded,
                                    filtered by group (e.g., "double_5fu"),
                                    and contains all necessary biomarker columns.
                            
            target_column (str): Name of the target variable column.
        """
        if not isinstance(df_main, pd.DataFrame):
            raise TypeError("df_main must be a pandas DataFrame.")
        if target_column not in df_main.columns:
            raise ValueError(f"Target column '{target_column}' not found in the provided DataFrame.")

        self.df = df_main.copy() # Work on a copy to avoid modifying original outside the class
        self.target_column = target_column
        self.X = None
        self.y = None
        self.variables = [] # List to store the names of feature variables used in the model

        # Initialize lists to accumulate performance metrics (precision, recall, f1-score, accuracy, AUC)
        # for both training and testing sets, across all seeds and folds.
        # Metrics for class 0 (e.g., control group)
        self.precision_class_0_train, self.recall_class_0_train, self.f1_class_0_train = [], [], []
        self.precision_class_1_train, self.recall_class_1_train, self.f1_class_1_train = [], [], []
        self.precision_class_0_test, self.recall_class_0_test, self.f1_class_0_test = [], [], []
        self.precision_class_1_test, self.recall_class_1_test, self.f1_class_1_test = [], [], []
        self.accuracy_train_all, self.accuracy_test_all = [], []
        self.auc_train_all, self.auc_test_all = [], []

        # Initialize cumulative confusion matrices for training and testing sets
        self.conf_matrix_train_total = np.zeros((2, 2))
        self.conf_matrix_test_total = np.zeros((2, 2))

        # List to store decision thresholds identified by the trees
        self.thresholds = []

        # Variables to store the first trained decision tree and its splitting feature for visualization
        self.primer_clf = None # To store the first trained tree for plotting
        self.primer_variable = None # To store the feature name for the first tree plot

        print("BiomarkerDecisionTree initialized with the provided DataFrame.")


    def set_features_and_target(self, feature_variables: list):
        """
        Sets the feature variables (X) and target variable (y) for the model.
        This method must be called before running cross-validation.

        Args:
            feature_variables (list): A list of column names to be used as features.
        """
        if not all(col in self.df.columns for col in feature_variables):
            missing_cols = [col for col in feature_variables if col not in self.df.columns]
            raise ValueError(f"One or more feature columns not found in the DataFrame: {missing_cols}")
        
        self.variables = feature_variables
        self.X = self.df[self.variables]
        self.y = self.df[self.target_column]
        print(f"Features set to: {self.variables}, Target set to: {self.target_column}")

    def _bootstrap_ci(self, data, num_iterations=10000, ci=95, random_state=42):
        """
        Calculates the percentile bootstrap confidence interval for a given metric's data.
        This is a helper method, indicated by the leading underscore.

        Args:
            data (list or np.array): The list or array of metric values (e.g., AUC scores from different seeds).
            num_iterations (int): The number of bootstrap samples to draw.
            ci (int): The confidence level for the interval (e.g., 95 for 95% CI).
            random_state (int): Seed for the random number generator to ensure reproducibility.

        Returns:
            tuple: A tuple containing the lower and upper bounds of the confidence interval.
        """
        np.random.seed(random_state) # Set seed for reproducibility of bootstrap sampling
        # Generate 'num_iterations' bootstrap samples by sampling with replacement
        boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_iterations)]
        # Calculate the percentile bounds for the confidence interval
        lower = np.percentile(boot_means, (100 - ci) / 2)
        upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
        return lower, upper

    def run_cross_validation(self, num_seeds=100, k_folds=5, max_depth=1):
        """
        Runs the stratified k-fold cross-validation procedure across multiple random seeds.
        This provides a more robust evaluation of the model's performance.

        Args:
            num_seeds (int): The number of independent repetitions (seeds) for the outer loop.
                             Each seed will generate a new set of train/test splits.
            k_folds (int): The number of folds for stratified K-Fold cross-validation within each seed.
                           Stratified ensures that each fold has approximately the same percentage
                           of samples of each target class as the complete set.
            max_depth (int): The maximum depth for the Decision Tree Classifier. A max_depth=1
                             creates a simple "decision stump".
        """
        # Reset all accumulated metrics before starting a new cross-validation run
        self.precision_class_0_train, self.recall_class_0_train, self.f1_class_0_train = [], [], []
        self.precision_class_1_train, self.recall_class_1_train, self.f1_class_1_train = [], [], []
        self.precision_class_0_test, self.recall_class_0_test, self.f1_class_0_test = [], [], []
        self.precision_class_1_test, self.recall_class_1_test, self.f1_class_1_test = [], [], []
        self.accuracy_train_all, self.accuracy_test_all = [], []
        self.auc_train_all, self.auc_test_all = [], []
        self.conf_matrix_train_total = np.zeros((2, 2))
        self.conf_matrix_test_total = np.zeros((2, 2))
        self.thresholds = [] # Also clear thresholds
        self.primer_clf = None
        self.primer_variable = None

        # Validate that features and target have been set
        if self.X is None or self.y is None:
            raise ValueError("Features and target not set. Call set_features_and_target() first.")
        if len(self.variables) > 1 and max_depth == 1:
            print("Warning: Decision Tree with max_depth=1 is typically used for a single feature. "
                  "If using multiple features, consider increasing max_depth or interpreting results carefully.")


        print(f"\nStarting cross-validation with {num_seeds} seeds and {k_folds}-fold cross-validation...")
        
        # Flag to ensure only the very first decision tree is saved for plotting
        primer_arbol_guardado = False

        # Outer loop: Iterates 'num_seeds' times, each time with a different random seed
        for seed in range(num_seeds):
            # Initialize Stratified K-Fold splitter with the current seed for reproducibility of splits
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

            # Lists to store metrics for the current seed (averaged across its 'k_folds')
            acc_train_fold, acc_test_fold = [], []
            auc_train_fold, auc_test_fold = [], []
            
            prec0_train_fold, rec0_train_fold, f10_train_fold = [], [], []
            prec1_train_fold, rec1_train_fold, f11_train_fold = [], [], []
            prec0_test_fold, rec0_test_fold, f10_test_fold = [], [], []
            prec1_test_fold, rec1_test_fold, f11_test_fold = [], [], []
            
            conf_matrix_train_seed = np.zeros((2, 2))
            conf_matrix_test_seed = np.zeros((2, 2))
            
            # Inner loop: Iterates through each fold of the current K-Fold split
            for fold_idx, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

                # Initialize and train the Decision Tree Classifier (a "stump" with max_depth=1)
                # 'class_weight='balanced'' handles potential class imbalance
                # 'random_state=seed' ensures reproducibility of the tree's internal randomness
                clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced', random_state=seed)
                clf.fit(X_train, y_train)

                # Store the first trained tree for plotting
                if not primer_arbol_guardado:
                    self.primer_clf = clf
                    # Determine the feature name used for splitting by the first node
                    # clf.tree_.feature[0] gives the index of the splitting feature
                    # -2 indicates a leaf node, meaning no split was made (e.g., if max_depth=0 or data is pure)
                    self.primer_variable = self.X.columns[clf.tree_.feature[0]] if clf.tree_.feature[0] != -2 else self.variables[0]
                    primer_arbol_guardado = True

                # Store the split threshold only if the tree actually made a split
                if clf.tree_.node_count > 1: # Check if there's at least one split
                    self.thresholds.append(clf.tree_.threshold[0])
                
                # Make predictions and predict probabilities for both training and testing sets
                y_train_pred = clf.predict(X_train)
                y_test_pred = clf.predict(X_test)
                y_train_prob = clf.predict_proba(X_train)[:, 1]
                y_test_prob = clf.predict_proba(X_test)[:, 1]

                # Generate classification reports (dictionaries) and confusion matrices
                report_train = classification_report(y_train, y_train_pred, output_dict=True, zero_division=0)
                report_test = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)

                cm_train = confusion_matrix(y_train, y_train_pred)
                cm_test = confusion_matrix(y_test, y_test_pred)

                # Store metrics for the training set (per fold)
                prec0_train_fold.append(report_train['0']['precision'])
                rec0_train_fold.append(report_train['0']['recall'])
                f10_train_fold.append(report_train['0']['f1-score'])

                prec1_train_fold.append(report_train['1']['precision'])
                rec1_train_fold.append(report_train['1']['recall'])
                f11_train_fold.append(report_train['1']['f1-score'])

                acc_train_fold.append(report_train['accuracy'])
                auc_train_fold.append(roc_auc_score(y_train, y_train_prob))
                conf_matrix_train_seed += cm_train

                # Store metrics for the test set (per fold)
                prec0_test_fold.append(report_test['0']['precision'])
                rec0_test_fold.append(report_test['0']['recall'])
                f10_test_fold.append(report_test['0']['f1-score'])

                prec1_test_fold.append(report_test['1']['precision'])
                rec1_test_fold.append(report_test['1']['recall'])
                f11_test_fold.append(report_test['1']['f1-score'])

                acc_test_fold.append(report_test['accuracy'])
                auc_test_fold.append(roc_auc_score(y_test, y_test_prob))
                conf_matrix_test_seed += cm_test

            # After all folds for a given seed are processed, average the metrics for this seed
            # and append to the global lists (which store one average value per seed).
            self.precision_class_0_train.append(np.mean(prec0_train_fold))
            self.recall_class_0_train.append(np.mean(rec0_train_fold))
            self.f1_class_0_train.append(np.mean(f10_train_fold))
            self.precision_class_1_train.append(np.mean(prec1_train_fold))
            self.recall_class_1_train.append(np.mean(rec1_train_fold))
            self.f1_class_1_train.append(np.mean(f11_train_fold))
            self.accuracy_train_all.append(np.mean(acc_train_fold))
            self.auc_train_all.append(np.mean(auc_train_fold))
            self.conf_matrix_train_total += conf_matrix_train_seed / k_folds

            self.precision_class_0_test.append(np.mean(prec0_test_fold))
            self.recall_class_0_test.append(np.mean(rec0_test_fold))
            self.f1_class_0_test.append(np.mean(f10_test_fold))
            self.precision_class_1_test.append(np.mean(prec1_test_fold))
            self.recall_class_1_test.append(np.mean(rec1_test_fold))
            self.f1_class_1_test.append(np.mean(f11_test_fold))
            self.accuracy_test_all.append(np.mean(acc_test_fold))
            self.auc_test_all.append(np.mean(auc_test_fold))
            self.conf_matrix_test_total += conf_matrix_test_seed / k_folds

        # After all seeds are processed, normalize the total confusion matrices by the number of seeds
        self.conf_matrix_train_total /= num_seeds
        self.conf_matrix_test_total /= num_seeds
        print("Cross-validation completed.")

    def get_seed_averaged_metric_scores(self, metric_name: str) -> np.ndarray:
        """
        Retrieves the NumPy array of averaged metric scores for each seed.
        These are the scores (averaged over k_folds) for each of the num_seeds runs.
        This method is useful for further analysis or plotting of metric distributions.

        Args:
            metric_name (str): The name of the metric to retrieve.
                               Supported names include: 'auc', 'accuracy', 'f1',
                               'f1_class_0', 'f1_class_1', 'precision_class_0',
                               'recall_class_0', 'precision_class_1', 'recall_class_1'.
                               Note: 'f1' specifically refers to 'f1_class_1_test' by default.

        Returns:
            np.ndarray: An array containing the metric scores, with one value per seed.

        Raises:
            ValueError: If cross-validation hasn't been run yet or if an invalid
                        metric name is provided.
        """
        metric_map = {
            'auc': self.auc_test_all,
            'accuracy': self.accuracy_test_all,
            'f1_class_0': self.f1_class_0_test,
            'f1_class_1': self.f1_class_1_test,
            'precision_class_0': self.precision_class_0_test,
            'recall_class_0': self.recall_class_0_test,
            'precision_class_1': self.precision_class_1_test,
            'recall_class_1': self.recall_class_1_test,
        }

        # Check if the requested metric name is supported
        if metric_name not in metric_map:
            raise ValueError(f"Unsupported metric name: '{metric_name}'. "
                             f"Supported metrics are: {list(metric_map.keys())}")

        scores = metric_map[metric_name]
        
        if not scores:
            raise ValueError(f"No scores available for metric '{metric_name}'. "
                             "Ensure run_cross_validation() has been executed.")
                             
        return np.array(scores)

    def print_results(self):
        """
        Prints the averaged performance metrics (mean and standard deviation)
        for both training and test sets, along with the average confusion matrices.
        """
        # Ensure cross-validation has been run before printing results
        if not self.auc_test_all:
            raise ValueError("Cross-validation not run. Call run_cross_validation() first.")

        print("\n--- Averaged Results ---")
        print("Average Confusion Matrix - Training:")
        print(self.conf_matrix_train_total)
        print("\nAverage Confusion Matrix - Test:")
        print(self.conf_matrix_test_total)

        # Print mean AUC-ROC for training and test sets
        print(f"\nAverage AUC-ROC Training: {np.mean(self.auc_train_all):.2f}")
        print(f"Average AUC-ROC Test: {np.mean(self.auc_test_all):.2f}")

        # Print average classification report for training set (Precision, Recall, F1 for both classes, and Accuracy)
        print("\nAverage Classification Report - Training:")
        print(f"Class 0 --- Precision: {np.mean(self.precision_class_0_train):.2f}, Recall: {np.mean(self.recall_class_0_train):.2f}, F1: {np.mean(self.f1_class_0_train):.2f}")
        print(f"Class 1 --- Precision: {np.mean(self.precision_class_1_train):.2f}, Recall: {np.mean(self.recall_class_1_train):.2f}, F1: {np.mean(self.f1_class_1_train):.2f}")
        print(f"Accuracy: {np.mean(self.accuracy_train_all):.2f}")

        # Print average classification report for test set
        print("\nAverage Classification Report - Test:")
        print(f"Class 0 --- Precision: {np.mean(self.precision_class_0_test):.2f}, Recall: {np.mean(self.recall_class_0_test):.2f}, F1: {np.mean(self.f1_class_0_test):.2f}")
        print(f"Class 1 --- Precision: {np.mean(self.precision_class_1_test):.2f}, Recall: {np.mean(self.recall_class_1_test):.2f}, F1: {np.mean(self.f1_class_1_test):.2f}")
        print(f"Accuracy: {np.mean(self.accuracy_test_all):.2f}")

        # Print standard deviations of metrics (across seeds), indicating variability in performance
        print("\n--- Metric Standard Deviations (across seeds) ---")
        print(f"\nAUC-ROC STD Training: {np.std(self.auc_train_all):.2f}")
        print(f"AUC-ROC STD Test: {np.std(self.auc_test_all):.2f}")
        print("\nStandard Deviation - Training:")
        print(f"Class 0 --- Precision: {np.std(self.precision_class_0_train):.2f}, Recall: {np.std(self.recall_class_0_train):.2f}, F1: {np.std(self.f1_class_0_train):.2f}")
        print(f"Class 1 --- Precision: {np.std(self.precision_class_1_train):.2f}, Recall: {np.std(self.recall_class_1_train):.2f}, F1: {np.std(self.f1_class_1_train):.2f}")
        print(f"Accuracy: {np.std(self.accuracy_train_all):.2f}")
        print("\nStandard Deviation - Test:")
        print(f"Class 0 --- Precision: {np.std(self.precision_class_0_test):.2f}, Recall: {np.std(self.recall_class_0_test):.2f}, F1: {np.std(self.f1_class_0_test):.2f}")
        print(f"Class 1 --- Precision: {np.std(self.precision_class_1_test):.2f}, Recall: {np.std(self.recall_class_1_test):.2f}, F1: {np.std(self.f1_class_1_test):.2f}")
        print(f"Accuracy: {np.std(self.accuracy_test_all):.2f}")

    def print_confidence_intervals(self, ci=95):
        """
        Prints the bootstrap percentile confidence intervals for key performance metrics
        on both training and test sets.

        Args:
            ci (int): The confidence level for the intervals (e.g., 95 for 95% CI).
        """
        # Ensure cross-validation has been run
        if not self.auc_test_all:
            raise ValueError("Cross-validation not run. Call run_cross_validation() first.")
        
        print(f"\n--- {ci}% Confidence Intervals (Bootstrap Percentile, Train) ---")
        ci_auc_train = self._bootstrap_ci(self.auc_train_all, ci=ci)
        print(f"AUC-ROC Train CI{ci}%: [{ci_auc_train[0]:.2f}, {ci_auc_train[1]:.2f}]")
        ci_acc_train = self._bootstrap_ci(self.accuracy_train_all, ci=ci)
        print(f"Accuracy Train CI{ci}%: [{ci_acc_train[0]:.2f}, {ci_acc_train[1]:.2f}]")
        ci_f1_0_train = self._bootstrap_ci(self.f1_class_0_train, ci=ci)
        print(f"F1 Score Class 0 Train CI{ci}%: [{ci_f1_0_train[0]:.2f}, {ci_f1_0_train[1]:.2f}]")
        ci_f1_1_train = self._bootstrap_ci(self.f1_class_1_train, ci=ci)
        print(f"F1 Score Class 1 Train CI{ci}%: [{ci_f1_1_train[0]:.2f}, {ci_f1_1_train[1]:.2f}]")
        ci_prec0_train = self._bootstrap_ci(self.precision_class_0_train, ci=ci)
        print(f"Precision Class 0 Train CI{ci}%: [{ci_prec0_train[0]:.2f}, {ci_prec0_train[1]:.2f}]")
        ci_rec0_train = self._bootstrap_ci(self.recall_class_0_train, ci=ci)
        print(f"Recall Class 0 Train CI{ci}%: [{ci_rec0_train[0]:.2f}, {ci_rec0_train[1]:.2f}]")
        ci_prec1_train = self._bootstrap_ci(self.precision_class_1_train, ci=ci)
        print(f"Precision Class 1 Train CI{ci}%: [{ci_prec1_train[0]:.2f}, {ci_prec1_train[1]:.2f}]")
        ci_rec1_train = self._bootstrap_ci(self.recall_class_1_train, ci=ci)
        print(f"Recall Class 1 Train CI{ci}%: [{ci_rec1_train[0]:.2f}, {ci_rec1_train[1]:.2f}]")

        print(f"\n--- {ci}% Confidence Intervals (Bootstrap Percentile, Test) ---")
        ci_auc_test = self._bootstrap_ci(self.auc_test_all, ci=ci)
        print(f"AUC-ROC Test CI{ci}%: [{ci_auc_test[0]:.2f}, {ci_auc_test[1]:.2f}]")
        ci_acc_test = self._bootstrap_ci(self.accuracy_test_all, ci=ci)
        print(f"Accuracy Test CI{ci}%: [{ci_acc_test[0]:.2f}, {ci_acc_test[1]:.2f}]")
        ci_f1_0_test = self._bootstrap_ci(self.f1_class_0_test, ci=ci)
        print(f"F1 Score Class 0 Test CI{ci}%: [{ci_f1_0_test[0]:.2f}, {ci_f1_0_test[1]:.2f}]")
        ci_f1_1_test = self._bootstrap_ci(self.f1_class_1_test, ci=ci)
        print(f"F1 Score Class 1 Test CI{ci}%: [{ci_f1_1_test[0]:.2f}, {ci_f1_1_test[1]:.2f}]")
        ci_prec0_test = self._bootstrap_ci(self.precision_class_0_test, ci=ci)
        print(f"Precision Class 0 Test CI{ci}%: [{ci_prec0_test[0]:.2f}, {ci_prec0_test[1]:.2f}]")
        ci_rec0_test = self._bootstrap_ci(self.recall_class_0_test, ci=ci)
        print(f"Recall Class 0 Test CI{ci}%: [{ci_rec0_test[0]:.2f}, {ci_rec0_test[1]:.2f}]")
        ci_prec1_test = self._bootstrap_ci(self.precision_class_1_test, ci=ci)
        print(f"Precision Class 1 Test CI{ci}%: [{ci_prec1_test[0]:.2f}, {ci_prec1_test[1]:.2f}]")
        ci_rec1_test = self._bootstrap_ci(self.recall_class_1_test, ci=ci)
        print(f"Recall Class 1 Test CI{ci}%: [{ci_rec1_test[0]:.2f}, {ci_rec1_test[1]:.2f}]")

    def plot_first_decision_tree(self):
        """
        Plots the very first Decision Tree trained during the cross-validation process.
        This provides a visual representation of the decision logic for one specific iteration.
        """
        if self.primer_clf is not None and self.primer_variable is not None:
            plt.figure(figsize=(8, 4))
            plot_tree(self.primer_clf, feature_names=[self.primer_variable], class_names=['0', '1'], filled=True)
            plt.title("Decision Tree (first trained model)")
            plt.savefig("decision_tree.tif", dpi=600)
            plt.show()
            print("Decision Tree plot saved as 'decision_tree.tif'")
        else:
            print("No decision tree was saved. Run run_cross_validation() first.")

    def analyze_thresholds(self):
        """
        Analyzes and plots the distribution of decision thresholds (split points)
        identified by the Decision Trees across all cross-validation runs.
        This helps understand the consistency of the optimal cutoff.
        """
        if not self.thresholds:
            print("No thresholds were recorded. Run run_cross_validation() first.")
            return
        
        # Calculate mean and standard deviation of recorded thresholds
        mean_threshold = np.mean(self.thresholds)
        std_threshold = np.std(self.thresholds)
        print(f"\nAverage decision threshold (cutoff line): {mean_threshold:.4f} ± {std_threshold:.4f}")
        # Calculate bootstrap confidence interval for the mean threshold
        ci_threshold = self._bootstrap_ci(self.thresholds)
        print(f"Cutoff (threshold) CI{95}%: [{ci_threshold[0]:.4f}, {ci_threshold[1]:.4f}]")

        # Plot the histogram of thresholds
        plt.figure(figsize=(8, 4))
        plt.hist(self.thresholds, bins=20, color='skyblue', edgecolor='black')
        plt.axvline(mean_threshold, color='red', linestyle='--', label=f'Average: {mean_threshold:.2f}')
        plt.xlabel(f'Cutoff value in {self.variables[0]}') # Assumes max_depth=1 usually means one primary feature
        plt.ylabel('Frequency')
        plt.title('Distribution of Decision Thresholds (Cutoff Lines)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout() # Adjust layout to prevent overlaps
        plt.show()
        print("Threshold distribution plot displayed.")

    def plot_metric_trends(self):
        """
        Plots the trend of various performance metrics (AUC-ROC, Accuracy, F1-scores,
        Precision, Recall) for both training and test sets across each of the
        cross-validation seeds (iterations). This helps visualize convergence and stability.
        """
        # Ensure metrics have been recorded
        if not self.auc_test_all:
            print("No metrics recorded. Run run_cross_validation() first.")
            return

        # Create a list of iteration numbers (from 0 to num_seeds-1)
        iterations = list(range(len(self.auc_train_all)))

        # Define the metrics to plot, along with their labels and plot titles
        metrics_to_plot = [
            (self.auc_train_all, self.auc_test_all, "AUC-ROC", "AUC-ROC per Seed"),
            (self.accuracy_train_all, self.accuracy_test_all, "Accuracy", "Accuracy per Seed"),
            (self.precision_class_0_train, self.precision_class_0_test, "Precision Class 0", "Precision Class 0 per Seed"),
            (self.recall_class_0_train, self.recall_class_0_test, "Recall Class 0", "Recall Class 0 per Seed"),
            (self.f1_class_0_train, self.f1_class_0_test, "F1 Class 0", "F1 Score Class 0 per Seed"),
            (self.precision_class_1_train, self.precision_class_1_test, "Precision Class 1", "Precision Class 1 per Seed"),
            (self.recall_class_1_train, self.recall_class_1_test, "Recall Class 1", "Recall Class 1 per Seed"),
            (self.f1_class_1_train, self.f1_class_1_test, "F1 Class 1", "F1 Score Class 1 per Seed")
        ]

        # Define grid dimensions for subplots
        n_rows = 4
        n_cols = 2
        
        # Create the figure and a grid of subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20)) # Adjust figsize for optimal clarity
        axes = axes.flatten() # Flatten the 2D array of axes for easier iteration

        # Loop through each metric and plot its trend on a dedicated subplot
        for i, (train_values, test_values, metric_name, title) in enumerate(metrics_to_plot):
            ax = axes[i] 
            # Plot training and testing values
            ax.plot(iterations, train_values, label=f'Train {metric_name}', color='blue', marker='o', linestyle='-')
            ax.plot(iterations, test_values, label=f'Test {metric_name}', color='red', marker='x', linestyle='--')
            ax.set_xlabel('Seed (Iteration)') # X-axis label
            ax.set_ylabel(metric_name)       # Y-axis label
            ax.set_title(title)              # Subplot title
            ax.legend()                      # Display legend for train/test lines
            ax.grid(True)                    # Add grid lines

        # Adjust layout to prevent overlapping titles/labels and add an overarching main title
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to make space for suptitle
        plt.suptitle("Metric Trends per Seed (Training vs. Test)", y=1.00, fontsize=18) # Main title for the entire figure
        plt.savefig("metric_trends_grid.tif", dpi=600) # Save the composite plot
        plt.show() # Display the plot
        print("Metric trend plots displayed in a 4x2 grid and saved as 'metric_trends_grid.tif'.")


