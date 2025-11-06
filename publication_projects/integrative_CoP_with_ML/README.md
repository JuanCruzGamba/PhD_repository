# Identification of Correlates of Protection for Vaccines against Trypanosoma cruzi using Machine Learning

This repository contains the code and datasets associated with the research work:

> Gamba et al., *Integrating Cellular Immune Biomarkers with Machine Learning to Identify Potential Correlates of Protection for a Trypanosoma cruzi Vaccine*.
> 
> Published: 28 August 2025.
> 
> Vaccines 2025, 13(9), 915. [DOI](https://doi.org/10.3390/vaccines13090915). Special Issue: Human Immune Responses to Infection and Vaccination

---

### 📂 Repository Contents:
- **Notebook**: Jupyter Notebook illustrating the complete analysis workflow, from exploratory analysis to model evaluation.
- **Code**: Python scripts to process biomarker data, perform biomarker engineering, train Decision Tree models, evaluate predictive performance with cross-validation, and perform the computational search for a potential integrative correlate of protection.
- **Data**: Preprocessed datasets for reproducibility.

This repository is designed to be accessible to researchers with little or no programming experience, particularly immunologists and biologists interested in learning how machine learning can be applied to immunological data.

All scripts include extensive comments and explanations, guiding the reader step by step through the data analysis, model building, and evaluation process. The goal is to make the computational workflow transparent, reproducible, and educational, bridging the gap between biological experimentation and data-driven modeling.

---

### 🚀 Quick Start Guide For Final Users (No Programming Experience Required)

#### Step 1: Get the Tool
1. Download this repository.
2. [Install Python 3.8+](https://www.python.org/downloads/) and [Jupyter Notebook or Jupyter Lab](https://jupyter.org/install).
3. Install required packages. The version of each package is listed below.

For detailed instructions on how to install packages using `pip`, please refer to the official documentation (https://packaging.python.org/en/latest/tutorials/installing-packages/)

#### Step 2: Get the Data
The analysis requires a `dataset.csv` file in the same directory as the notebook. You can:
- **Use the provided file**: Already included in this repository, or
- **Download from Zenodo**: (https://zenodo.org/records/16281869)

#### Step 3: Run the Analysis
1. Open `notebook_CoPs_ML_v1.ipynb` in Jupyter Notebook or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JuanCruzGamba/PhD_repository/blob/main/publication_projects/integrative_CoP_with_ML/notebook_CoPs_ML_v1.ipynb) (remember also upload `dataset.csv` to Colab)

2. Execute all cells step by step.
3. Get your **pICoP scores** and protection predictions!

#### Step 4: Interpret Your Results
- **pICoP Score:** Your integrated biomarker
- **Threshold:** Values ≥30.57 suggest higher survival probability
- **Performance Metrics:** How well the model works with your data

### For Experienced Users
1. Clone repository: `git clone [repository-url]`
2. Ensure `dataset.csv` is in the working directory (included in repo or download from Zenodo)
3. Install dependencies.
4. Run analysis.

---

### 💻 Repository Code Structure

This repository includes two main Python scripts and one Jupyter Notebook, each with specific roles in the analysis workflow:

- **`notebook_CoPs_ML.ipynb`**  
  The notebook orchestrates the overall analysis by loading the processed data, invoking the model defined in `biomarker_decision_tree.py`, and running the evaluation workflow. It generates visualizations and summarizes results in an accessible and reproducible format. Ensure that `biomarker_decision_tree.py` is located in the same directory or accessible in the Python path for proper import.

- **`biomarker_decision_tree.py`**  
  This script defines the `BiomarkerDecisionTree` class, encapsulating the machine learning pipeline for training, validating, and evaluating Decision Tree models. It includes the implementation of repeated stratified k-fold cross-validation (5 folds, 100 repetitions with different random seeds), metric calculations (Accuracy, AUC-ROC, F1-score, Precision, Recall), and performance aggregation. The encapsulation facilitates code reuse and modular analysis.
  
- **`pICoP_search.py`**  
  This script implements a systematic computational search to identify the optimal weighted linear combination of cellular immune biomarkers (%CD8⁺, %CD4⁺, and %CD11b⁺Gr-1⁺ MDSC-like cells). It exhaustively evaluates all possible weight combinations within a specified range using a simple decision stump model. The resulting composite biomarker, named **pICoP** (2 × %CD8⁺ + %CD4⁺ − %CD11b⁺Gr-1⁺), was selected for its superior predictive performance (high F1-Score and AUC-ROC), simplicity, and statistical significance confirmed by a permutation test (p < 0.05). This approach validates the initial biologically driven design.

---

### 📊 Data Availability
The dataset is publicly available at:
- [Zenodo DOI: 10.5281/zenodo.16281869](https://doi.org/10.5281/zenodo.16281869)  
- Included in this repository.

---

### ⚙️ Requirements
Main package versions used:
- Python 3.12.1  
- NumPy 2.0.2  
- Pandas 2.2.3  
- Scikit-learn 1.5.2  
- Matplotlib 3.9.3
- Joblib 1.4.2
- tqdm 4.67.1  

---

### 📌 Background
Chagas disease, caused by the protozoan parasite *Trypanosoma cruzi*, remains a major public health concern in Latin America, with no licensed vaccine available. Identifying **correlates of protection (CoPs)** — measurable immune parameters that predict vaccine efficacy — can accelerate vaccine development. While most known CoPs are antibody-based, they have not been validated in *T. cruzi* vaccine research. This work explores the use of **cellular immune biomarkers** as alternative CoPs.

---

### 🧪 Study Overview
- **Experimental model**: Mice immunized with a trans-sialidase (TSf)-based vaccine candidate, potentiated with 5-fluorouracil (5FU) to deplete myeloid-derived suppressor cells (MDSCs).
- **Biomarker acquisition**: Percentages of CD4⁺, CD8⁺, and CD11b⁺Gr-1⁺ cells measured by flow cytometry from peripheral blood.
- **Outcome**: Survival at day 25 post-*T. cruzi* challenge.
- **Analysis approach**: Machine Learning (ML) with Decision Tree models to identify and evaluate potential CoPs.

---

### 📊 Study Workflow

![Workflow Figure](images/figure_workflow.png)

**Workflow including experimental timeline, dataset construction, and machine learning analysis.**  
The study can be divided into three steps:  
A) **Experimental timeline** – Female BALB/c mice were assigned to control (PBS) or vaccinated (double 5FU TSf‐ISPA) groups. The vaccinated group received three subcutaneous doses of TSf‐ISPA (on days 0, 15, and 30), each preceded and followed by intraperitoneal 5FU administration to reduce MDSC expansion. Flow cytometry of peripheral blood was performed 48 hours after the final TSf‐ISPA dose. All groups were intraperitoneally infected with 1600–1700 Tulahuen strain trypomastigotes. Survival was recorded until day 35 p.i. The experiment was repeated three times to collect a sufficient number of animals.  
B) **Dataset construction** – Flow cytometry data and corresponding survival outcomes for each mouse at day 25 p.i. were compiled into a dataset combining the three assays performed.  
C) **Machine learning analysis** – The dataset was used to train and test a Decision Tree classification model to identify potential CoPs associated with survival.

---

### 🔍 Main Findings
- Individual biomarkers showed limited predictive power.
- **Biomarker engineering** combining effector and regulatory arms of the immune response improved prediction:
  - **REB** = (%CD8⁺ + %CD4⁺) − %CD11b⁺Gr-1⁺
  - **pICoP** (Potential Integrative CoP) = 2 × %CD8⁺ + %CD4⁺ − %CD11b⁺Gr-1⁺
- The pICoP significantly enhanced a simple one-level Decision Tree’s predictive performance:
  - **Accuracy**: ~0.86
  - **AUC-ROC**: ~0.87

---

### 🧮 Machine Learning Workflow: Decision Tree with Repeated Stratified k-Fold Cross-Validation

The analysis implemented in this repository follows a **nested repetition–fold structure** to ensure robust performance estimation and minimize variance due to random data partitioning.

**Step-by-step process:**

1. **Feature and target definition**  
   The model uses selected biomarker variables (e.g., %CD4⁺, %CD8⁺, %CD11b⁺Gr-1⁺, engineered biomarker like pICoP) as features (**X**) and survival status at day 25 p.i. as the target (**y**).

2. **Outer loop: Seed iterations**  
   - The procedure is repeated for a specified number of random seeds (e.g., 100 iterations).  
   - Each seed defines a unique random split pattern for the cross-validation folds.

3. **Inner loop: Stratified k-Fold cross-validation**  
   - For each seed, a **StratifiedKFold** split is performed (e.g., 5 folds) to preserve the class distribution in each fold.  
   - The model is trained and evaluated **k** times per seed, each time using a different fold as the test set.

4. **Model training and prediction per fold**  
   - A **Decision Tree Classifier** with `max_depth=1` (decision stump) and `class_weight='balanced'` is trained on the training set.  
   - Predictions (class labels and probabilities) are generated for both the training and test sets.

5. **Metric computation per fold**  
   For each fold, the following metrics are calculated **per class** and stored:
   - Precision, Recall, and F1-score for class 0 and class 1.
   - Accuracy.
   - Area Under the ROC Curve (AUC).  
   Confusion matrices are also aggregated per seed.

6. **Per-seed averaging**  
   - Metrics from the **k folds** are averaged to obtain a **single performance value per metric for that seed**.  
   - Confusion matrices are normalized across folds.

7. **Global aggregation**  
   - After all seeds are processed, the global performance distribution for each metric is obtained (one value per seed).  
   - Metrics can then be summarized as:
     - **Mean ± standard deviation** across seeds.
     - **95% confidence intervals** using bootstrap resampling.

**Advantages of this approach:**
- Reduces overfitting risk by repeatedly evaluating on unseen data.
- Produces **stable performance estimates** less dependent on a single split.
- Allows for **statistical characterization** of metric variability.

---

### 📜 Citation
If you use this code or data, please cite our work:  
Gamba, J. C., Borgna, E., Prochetto, E., Pérez, A. R., Batista-Duharte, A., Marcipar, I., Gerard, M., & Cabrera, G. (2025). Integrating Cellular Immune Biomarkers with Machine Learning to Identify Potential Correlates of Protection for a Trypanosoma cruzi Vaccine. Vaccines, 13(9), 915. https://doi.org/10.3390/vaccines13090915 

--- 

**⚠️ Disclaimer:** This tool is for research purposes only. Results should be validated in appropriate experimental systems before application. The authors are not responsible for misuse or misinterpretation of results.
