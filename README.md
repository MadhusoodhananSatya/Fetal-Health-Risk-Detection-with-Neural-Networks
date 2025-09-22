🌟 Cardiotocographic Classification
===================================

This project applies and compares **deep learning (Neural Networks)** and **traditional machine learning algorithms (Logistic Regression, Random Forest)** to classify fetal cardiotocographic (CTG) readings into three categories of fetal health: **Normal, Suspect, and Pathological**. The goal is to evaluate which modeling approach provides the most reliable predictions while also exploring model interpretability, an essential aspect in healthcare applications.

The pipeline begins with **exploratory data analysis (EDA)** to understand the dataset, followed by **data preprocessing** (label transformation, scaling, and class imbalance handling). A **neural network model** is trained with class weighting, while Logistic Regression and Random Forest serve as benchmarks.

The project emphasizes not only prediction accuracy but also **interpretability**, using Random Forest feature importances and SHAP (SHapley Additive exPlanations) to explain model behavior.

* * * * *

✨ Key Features & Technical Details
----------------------------------

**Exploratory Data Analysis (EDA):**

-   Dataset inspection using `.head()`, `.info()`, `.describe()`, and `.shape()`.

-   Class distribution analysis of `NSP` (Normal, Suspect, Pathological), visualized with count plots and pie charts, showing a clear imbalance in favor of Normal cases.

-   Correlation heatmap of numerical features to identify interdependencies.

-   Violin and box plots of key features (`BPM`, `ASTV`, `MSTV`, `Max`) across different classes.

**Data Preprocessing:**

-   Transformed target variable: `NSP` shifted from {1, 2, 3} → {0, 1, 2}.

-   Stratified train-test split (80/20) to maintain class balance across sets.

-   Applied **StandardScaler** to normalize features for consistent input ranges.

**Class Imbalance Mitigation:**

-   Instead of oversampling, applied **class weighting** during neural network training:

    -   Normal (0): weight = 1

    -   Suspect (1): weight = 5.6

    -   Pathological (2): weight = 9.4

**Feature Engineering & Scaling:**

-   Retained all numerical features after correlation and distribution checks.

-   Standardized inputs for both neural networks and classical ML models.

**Model Implementation & Hyperparameter Details:**

-   **Neural Network (TensorFlow/Keras):**

    -   Architecture: Dense(15, ReLU) → Dense(10, Softmax).

    -   Optimizer: Adam. Loss: Sparse categorical cross-entropy.

    -   Trained for 200 epochs, batch size = 32.

    -   Validation split of 20%.

    -   Trained with class weights to handle imbalance.

-   **Logistic Regression:**

    -   Configured with `max_iter=1000`.

    -   Provides baseline linear separability performance.

-   **Random Forest Classifier:**

    -   200 estimators, random_state = 0.

    -   Extracted top 10 feature importances for interpretability.

    -   Applied SHAP for feature-level explanations across predictions.

**Model Evaluation:**

-   Neural Network performance tracked with training and validation **accuracy/loss curves**.

-   **ROC curves and AUC values** computed per class for multi-class evaluation.

-   **Normalized confusion matrix** visualized classification outcomes.

-   **Classification report** generated with accuracy, precision, recall, and F1-scores for each class.

-   Random Forest feature importance bar chart and SHAP summary plot provided interpretability insights.

-   PCA visualization projected the dataset into 2D, showing class clusters.

* * * * *

🚀 Getting Started
------------------

To run this project, you will need a Python environment with the following libraries:

-   pandas

-   numpy

-   matplotlib

-   seaborn

-   scikit-learn

-   shap

-   tensorflow

You can set up the environment and run the analysis by cloning the repository and executing **Cardiotocographic_analysis.ipynb** in a Jupyter Notebook environment.

* * * * *

📊 Project Workflow
-------------------

The **Cardiotocographic_analysis.ipynb** notebook follows a structured machine learning workflow:

**Data Loading & Inspection:**

-   Import required libraries.

-   Load `Cardiotocographic.csv` into a pandas DataFrame.

-   Inspect dataset structure with `.info()`, `.shape()`, `.describe()`.

**Data Cleaning & Exploration:**

-   Confirm no missing values or duplicates.

-   Visualize class imbalance using count plots and pie charts.

-   Explore distributions of important features (`BPM`, `ASTV`, `MSTV`, `Max`).

-   Generate correlation heatmap of numerical features.

**Data Preparation:**

-   Encode labels (`NSP - 1`).

-   Split into training and testing sets with stratification.

-   Standardize features with **StandardScaler**.

-   Assign class weights to address imbalance.

**Model Building & Training:**

-   **Neural Network:** Defined, compiled, and trained with class weighting for 200 epochs.

-   **Logistic Regression & Random Forest:** Trained as benchmark models.

**Model Evaluation:**

-   Compare training vs validation accuracy/loss curves (Neural Network).

-   Plot ROC curves with AUC values for each class.

-   Display confusion matrix heatmaps (normalized and raw).

-   Generate classification report with precision, recall, F1-score.

-   Feature importance analysis with Random Forest.

-   Interpret results with SHAP summary plots.

**Visualization & Dimensionality Reduction:**

-   PCA used to reduce dimensionality and visualize class clusters in 2D.

* * * * *

📈 Final Thoughts
-----------------

The analysis demonstrates the potential of machine learning in predicting fetal health status from cardiotocographic data:

-   The **Neural Network**, trained with class weights, achieved robust performance across all classes and was particularly effective at mitigating imbalance bias.

-   **Random Forest** offered competitive accuracy while also providing interpretability via feature importances and SHAP values, highlighting the influence of variability measures (e.g., ASTV, MSTV).

-   **Logistic Regression** served as a strong baseline but was outperformed by both the neural network and Random Forest.

The project illustrates the trade-off between **accuracy (Neural Network)** and **interpretability (Random Forest)** --- a critical consideration in healthcare applications where decision transparency is essential.

Future directions could include:

-   Applying oversampling techniques like SMOTE to complement class weighting.

-   Hyperparameter tuning of the neural network (dropout, deeper architectures).

-   Testing gradient boosting algorithms such as XGBoost or LightGBM.

* * * * *

🙏 Acknowledgments
------------------

I extend my thanks to the creators of **TensorFlow, scikit-learn, shap, pandas, matplotlib, and seaborn** for the powerful tools that made this analysis possible.
