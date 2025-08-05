## Machine Learning to Integrate Cellular Immune Biomarkers to Identify Potential Correlates of Protection for a *Trypanosoma cruzi* Vaccine

This repository contains the code and datasets associated with the research work:

> Gamba et al., *Integrating Cellular Immune Biomarkers with Machine Learning to Identify Potential Correlates of Protection for a Trypanosoma cruzi Vaccine*.

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

### 📂 Repository Contents
- **Code**: Python scripts to process biomarker data, perform biomarker engineering, train Decision Tree models, and evaluate predictive performance with cross-validation.
- **Data**: Preprocessed datasets for reproducibility.
- **Notebooks**: Jupyter Notebooks illustrating the complete analysis workflow.

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

---

### 📜 Citation
If you use this code or data, please cite our work:  
Gamba et al., *Integrating Cellular Immune Biomarkers with Machine Learning to Identify Potential Correlates of Protection for a Trypanosoma cruzi Vaccine*. 


