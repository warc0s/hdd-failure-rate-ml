# HDD Failure Prediction using Machine Learning

---

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Removing Irrelevant Columns](#removing-irrelevant-columns)
  - [Selecting SMART Attributes](#selecting-smart-attributes)
  - [Normalizing Capacity](#normalizing-capacity)
- [Feature Engineering](#feature-engineering)
  - [Converting Model to Manufacturer](#converting-model-to-manufacturer)
  - [Encoding Manufacturer Information](#encoding-manufacturer-information)
- [Model Training](#model-training)
  - [Random Forest Classifier](#random-forest-classifier)
  - [XGBoost Classifier](#xgboost-classifier)
- [Model Evaluation](#model-evaluation)
  - [Confusion Matrix](#confusion-matrix)
  - [Classification Report](#classification-report)
  - [Feature Importance](#feature-importance)
  - [Threshold Adjustment](#threshold-adjustment)
- [Model Ensembling](#model-ensembling)
- [Implementation and Usage](#implementation-and-usage)
- [Conclusions and Future Work](#conclusions-and-future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Introduction

Predicting hard disk drive (HDD) failures is crucial for maintaining data integrity and minimizing downtime in data centers. This project aims to develop a binary classification model using machine learning algorithms, primarily focusing on Random Forest and XGBoost, to predict the probability of HDD failures based on SMART (Self-Monitoring, Analysis, and Reporting Technology) data and other key features such as brand and storage capacity.

---

## Data Collection

The dataset used in this project is provided by [Backblaze](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data), which offers detailed information about the HDDs deployed in their data centers. The data includes daily reports aggregated quarterly, containing:

- **Date**: When the data was recorded (yyyy-mm-dd).
- **Serial Number**: Manufacturer-assigned serial number.
- **Model**: Manufacturer-assigned model number.
- **Capacity**: Storage capacity in bytes.
- **Failure**: Binary indicator (0 for operational, 1 for failure).
- **SMART Attributes**: 124 columns representing raw and normalized values across 62 different SMART statistics.

### Step 1: Data Acquisition

1. **Download and Extract Data**: 
   - Download ZIP files containing the first two quarters of 2024 data.
   - Extract the CSV files into a single directory for easier access.

2. **Inspect Columns**:
   - Given the extensive number of columns, display them in a horizontal format for better readability.

---

## Handling Class Imbalance

The dataset exhibits a significant class imbalance, with approximately 0.75% of the HDDs failing. Specifically, daily failures range between 8-13 out of over 280,000 operational drives.

### Step 2.1: Daily Failures Review

- **Random Sampling**: Examine 5 random CSV files to determine the number of HDD failures each day.
- **Data Cleaning**: Remove non-informative `.csv` files that contain metadata to prevent errors during processing.

### Step 2.2: Aggregating Failures

- **Script Development**: Create a script to iterate through 183 CSV files, extracting only the rows where HDDs failed.
- **Dataset Consolidation**: Combine these failure records with the latest CSV (184th) to form a more balanced dataset.

### Step 2.3: Merging CSVs

- **Final Dataset**: Merge `CSV_fallos.csv` with the June 30th CSV to create `dataset.csv`.
- **Shuffle and Relocate**: Shuffle the combined dataset and move it to the root directory for ease of access.

**Result**: Approximately 2,200 failed HDDs (~0.75% of the total dataset).

---

## Data Cleaning and Preprocessing

A comprehensive dataset requires thorough cleaning to ensure reliability and effectiveness in training the machine learning models.

### Step 3.1: Removing Irrelevant Columns

Eliminate columns that do not contribute to predicting HDD failures:

- **Non-Predictive Columns**:
  - `serial_number`
  - `datacenter`
  - `cluster_id`
  - `vault_id`
  - `pod_id`
  - `pod_slot_num`
  - `is_legacy_format`
  - `date`

- **SMART Raw Values**: Remove unnormalized SMART attributes, retaining only the normalized values for consistency and to reduce manufacturer dependency.

### Step 3.2: Selecting SMART Attributes

- **Missing Values Analysis**:
  - Evaluate the percentage of missing values for each normalized SMART attribute.
  - Retain attributes with ≤5-10% missing data.
  - Discard rows with any missing values in the selected SMART attributes to avoid introducing noise.

**Selected SMART Attributes** (13 total):

1. **smart_1**: Read error rate.
2. **smart_3**: Spin-up time.
3. **smart_4**: Start/stop count.
4. **smart_5**: Reallocated sectors count.
5. **smart_7**: Seek error rate.
6. **smart_9**: Power-on hours.
7. **smart_10**: Spin retry count.
8. **smart_12**: Power cycle count.
9. **smart_192**: Power-off retract count.
10. **smart_193**: Load cycle count.
11. **smart_197**: Current pending sector count.
12. **smart_198**: Offline uncorrectable sector count.
13. **smart_199**: UltraDMA CRC error count.

### Step 3.3: Normalizing Capacity

- **Issue**: The `capacity_bytes` column contains large values that could disproportionately influence the model.
- **Solution**:
  - Convert `capacity_bytes` to terabytes by dividing by \(1 \times 10^{12}\) and rounding to two decimal places.
  - Create a new column `capacity_terabytes` and remove the original `capacity_bytes`.

---

## Feature Engineering

Enhancing the dataset with meaningful features can significantly improve model performance.

### Step 5: Converting Model to Manufacturer

The `model` column represents the HDD model, which could provide insights into the longevity of different HDD models. However, using the model directly has drawbacks:

- **Pros**:
  - Potentially better survival rates by model.
  
- **Cons**:
  - Limited utility for predicting HDD models not present in the training set.

**Solution**: Implement a hybrid approach by extracting the `manufacturer` from the `model` using a Large Language Model (LLM) like Llama 3.1 70B. This allows the model to generalize better across different HDD models by focusing on the manufacturer.

### Step 5.2: Automating Manufacturer Extraction

- **Approach**:
  - Use the LLM to map each `model` to its corresponding `manufacturer`.
  - Store mappings in a dictionary to avoid redundant API calls and improve efficiency.
  - Assign a generic manufacturer label for unrecognized models.

### Step 6: Encoding Manufacturer Information

Convert the `manufacturer` categorical data into numerical format to be usable by machine learning algorithms.

- **One-Hot Encoding**:
  - Create binary columns for each manufacturer.
  - Advantages:
    - No implicit ordering.
    - Treats manufacturers as independent categories.
  - Allows for the inclusion of unknown manufacturers by setting all manufacturer columns to 0.

---

## Model Training

With a cleaned and preprocessed dataset, the next step is to train machine learning models to predict HDD failures.

### Step 7.1: Random Forest Classifier

- **Dataset Split**:
  - 90% for training (~254,700 samples).
  - 10% for testing (~28,300 samples).
  - Ensures class proportion is maintained in both sets.

- **Hyperparameter Tuning**:
  - **GridSearchCV** (commented out for local execution):
    - `n_estimators`: [100, 200, 300]
    - `max_depth`: [10, 20, 30]
    - `min_samples_split`, `min_samples_leaf`: Optimized for generalization.
    - `max_features`: ['sqrt', 'log2']
  - **Best Parameters**: Manually assigned post GridSearchCV due to computational constraints.

- **Objective**: Maximize the F1-Score for the minority class to balance precision and recall.

### Step 7.2: XGBoost Classifier

- **Dataset Split**:
  - Consistent with Random Forest for fair comparison.

- **Hyperparameter Tuning**:
  - **GridSearchCV** (commented out for local execution):
    - `n_estimators`: [100, 200, 300]
    - `max_depth`: [3, 6, 10]
    - `learning_rate`: [0.01, 0.1, 0.2]
    - `subsample`, `colsample_bytree`: To enhance model robustness.

- **Objective**: Utilize gradient boosting to handle complex patterns and class imbalance effectively.

---

## Model Evaluation

Evaluating the performance of trained models is critical to understanding their effectiveness and suitability for deployment.

### Step 8.1: Confusion Matrix

Visualizes the performance of both models by showing the true positives, true negatives, false positives, and false negatives.

### Step 8.2: Classification Report

Provides detailed metrics including precision, recall, and F1-Score for each class, offering deeper insights into model performance.

### Step 8.3: Feature Importance

Analyzes which features are most influential in predicting HDD failures for each model.

- **Random Forest**:
  - Focuses on `smart_9` (Power-on hours) and `smart_1` (Read error rate).

- **XGBoost**:
  - Prioritizes `smart_197` (Current pending sector count).

This indicates that Random Forest emphasizes the age of the HDD, while XGBoost focuses more on physical failure indicators.

### Step 8.4: Threshold Adjustment

Adjusts the decision threshold from the default 50% to optimize the F1-Score and recall.

- **Approach**:
  - Test thresholds from 10% to 90% in 1% increments.
  - Select the threshold that maximizes the F1-Score, with a preference for higher recall in case of ties.

- **Results**:
  - **Random Forest**: Improved F1-Score by 11 points.
  - **XGBoost**: Achieved a higher F1-Score compared to Random Forest.

---

## Model Ensembling

Combining the strengths of both Random Forest and XGBoost models through weighted ensemble stacking to enhance prediction performance.

- **Method**:
  - Assign weights to each model’s predictions.
  - Experiment with various weight combinations and thresholds.
  - Identify the best combinations to maximize F1-Score, precision, and recall.

- **Best Combinations**:
  - **Maximize F1-Score**: XGBoost (0.56), Random Forest (0.44), Threshold: 25%
  - **Maximize Precision**: XGBoost (0.47), Random Forest (0.53), Threshold: 88%
  - **Maximize Recall**: XGBoost (0.04), Random Forest (0.96), Threshold: 10%

These combinations provide a balanced approach to handling different aspects of model performance, allowing for a more robust final prediction system.

---

## Implementation and Usage

The final implementation leverages the ensemble of both models with optimized weights and thresholds to predict HDD failures. The process involves:

1. **Selecting Disks**: Randomly select HDDs from the test set, ensuring at least one has failed.
2. **Applying Ensemble Logic**:
   - **F1-Oriented**: High accuracy in predicting failures.
   - **Precision-Oriented**: Identifies critical failures with high precision.
   - **Recall-Oriented**: Maximizes the detection of actual failures, albeit with some false positives.
3. **Decision Making**:
   - Utilize a series of conditional checks to determine the final prediction based on the ensemble outputs.
   - Provide actionable alerts based on the combination of model predictions.

**Usage Example**:
Run the provided script to evaluate randomly selected HDDs and observe the ensemble-based predictions, validating the model's real-world applicability.

---

## Conclusions and Future Work

### Key Takeaways

- **Ensemble Approach**: Combining Random Forest and XGBoost leverages their individual strengths, resulting in a more robust prediction system.
- **Threshold Optimization**: Adjusting decision thresholds significantly improves the model’s ability to balance precision and recall.
- **Feature Selection**: Focusing on critical SMART attributes enhances model performance by reducing noise.

### Areas for Improvement

1. **Expand Failure Data**: Incorporate historical failure data from previous years to increase the minority class representation.
2. **Implement SMOTE**: Apply Synthetic Minority Over-sampling Technique to further address class imbalance.
3. **Temporal Features**: Analyze SMART attributes over a time window (e.g., last 10 days) to capture trends leading to failure.
4. **Re-evaluate SMART Attributes**: Investigate additional SMART attributes with low missing rates for potential inclusion.
5. **Add More Models**: Integrate additional algorithms like LightGBM or SVM to capture different data patterns.
6. **Train Meta-Model**: Develop a meta-model that learns from the predictions of individual models for enhanced accuracy.
7. **Advanced Neural Networks**: Explore deep learning architectures to model complex relationships between features.
8. **Model-Specific Training**: Train separate models for different HDD models to capture model-specific failure patterns.
9. **One-Hot Encoding for Models**: Instead of manufacturers, encode HDD models to preserve detailed information.

### Final Thoughts

This project demonstrates the potential of machine learning in predicting HDD failures, offering a foundation for more advanced predictive maintenance systems. By implementing the suggested improvements, the model's accuracy and reliability can be further enhanced, making it a valuable tool for data center operations.

---

## Acknowledgements

- **Backblaze**: For providing the comprehensive HDD failure dataset. You can check it [here](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).
- **Groq**: For facilitating access to Llama 3.1 70B for manufacturer extraction.

---

## License

This project is licensed under the Apache2 License.

---

*Thank you for taking the time to explore this project! Feel free to reach out with any questions or suggestions.*
