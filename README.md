# HDD Failure Prediction using Machine Learning

---

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
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

---

## Notes

1. In this README, I will provide a quick overview of the steps followed to achieve the best machine learning model for this project. For a more detailed explanation or to review the code used, please refer to the accompanying Jupyter notebook (`.ipynb`).
2. The RandomForest model is not included here as its file size exceeds 25MB. If anyone would like to access it, they can either run the training code provided or request it from me via email or LinkedIn.
3. There is also a 10-minute video in Spanish explaining this project. However, please note that the values shown in the video are different, as it was based on the initial trial. The models and code in this repository are from the second version, with improved values. [Project Explanation Video (Spanish)](https://www.youtube.com/watch?v=jqhft3HSQkY)


---

### Step 1: Data Acquisition

1. **Download and Extract Data**: 
   - Download ZIP files containing data from 2018 to the first two quarters of 2024.
   - Extract the CSV files into a single directory for easier access.

2. **Inspect Columns**:
   - Given the extensive number of columns, display them in a horizontal format for better readability.

---

## Handling Class Imbalance

The dataset exhibits a significant class imbalance, as selecting a random day's CSV file for training would show that only about 5-10 HDDs fail each day, out of a large number of operational drives. This means that on any given day, failures are relatively rare, leading to a highly imbalanced dataset.

### Step 2.1: Daily Failures Review

- **Random Sampling**: Examine 5 random CSV files to determine the number of HDD failures each day.
- **Data Cleaning**: Remove non-informative `.csv` files that contain metadata to prevent errors during processing.

### Step 2.2: Aggregating Failures

- **Script Development**: Create a script to iterate through all CSV files, extracting only the rows where HDDs failed.
- **Dataset Consolidation**: Combine these failure records with the latest CSV from June 30 to form a more balanced dataset.

### Step 2.3: Merging CSVs

- **Final Dataset**: Merge `CSV_fallos.csv` with the June 30th CSV to create `dataset.csv`.
- **Shuffle and Relocate**: Shuffle the combined dataset and move it to the root directory for ease of access.

**Result**: Approximately 17,000 failed HDDs (~5.54% of the total dataset).

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

### Step 4: Normalizing Capacity

- **Issue**: The `capacity_bytes` column contains large values that could disproportionately influence the model.
- **Solution**:
  - Convert `capacity_bytes` to terabytes by dividing by 10^12 and rounding to two decimal places.
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

**Solution**: Implement a hybrid approach by extracting the `manufacturer` from the `model` using a Large Language Model (LLM) like Llama 3.1 70B (thanks to Groq for its free tier, which made this possible). This allows the model to generalize better across different HDDs by focusing on the manufacturer, enabling a broader scope than using specific model details.

### Step 5.2: Automating Manufacturer Extraction

- **Approach**:
  - Use the LLM to map each `model` to its corresponding `manufacturer`.
  - Store mappings in a dictionary to avoid redundant API calls and improve efficiency.
  - Assign a generic manufacturer label for unrecognized models.

> **Note**: This step was ultimately unnecessary, as the LLM (Llama 3.1 70B) successfully extracted the manufacturer for all HDDs.

### Step 6: Encoding Manufacturer Information

Convert the `manufacturer` categorical data into a numerical format to make it usable by machine learning algorithms.

- **One-Hot Encoding**:
  - Transform each manufacturer into a binary column.
  - Advantages:
    - Avoids imposing any order on manufacturers.
    - Treats each manufacturer as an independent category.
  - Chose this method for its flexibility, allowing unknown manufacturers to be represented by setting all columns to 0.

---

## Model Training

With a cleaned and preprocessed dataset, the next step is to train machine learning models to predict HDD failures.

### Step 7.1: Random Forest Classifier

- **Dataset Split**:
  - 90% for training (~254,700 samples).
  - 10% for testing (~28,300 samples).
  - Ensures class proportion is maintained in both sets.

- **Hyperparameter Tuning**:
  - **GridSearchCV**:
    - `n_estimators`: [100, 200, 300]
    - `max_depth`: [10, 20, 30]
    - `min_samples_split`, `min_samples_leaf`: Optimized for generalization.
    - `max_features`: ['sqrt', 'log2']

- **Objective**: Maximize the F1-Score for the minority class to balance precision and recall.

### Step 7.2: XGBoost Classifier

- **Dataset Split**:
  - Consistent with Random Forest for fair comparison.

- **Hyperparameter Tuning**:
  - **GridSearchCV**:
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

### Classification Report for XGBoost:

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| No Failure  | 0.98      | 0.99   | 0.99     | 27960   |
| Failure     | 0.88      | 0.70   | 0.78     | 1630    |
| Accuracy |         |        | 0.98 | 29590   |
| Macro Avg | 0.93      | 0.85   | 0.89     | 29590   |
| Weighted Avg | 0.98   | 0.98   | 0.98     | 29590   |

---

### Classification Report for RandomForest:

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| No Failure  | 0.98      | 0.99   | 0.99     | 27960   |
| Failure     | 0.88      | 0.70   | 0.78     | 1630    |
| Accuracy |         |        | 0.98 | 29590   |
| Macro Avg | 0.93      | 0.85   | 0.89     | 29590   |
| Weighted Avg | 0.98   | 0.98   | 0.98     | 29590   |

### Step 8.3: Feature Importance

Analyzes which features are most influential in predicting HDD failures for each model.

- **Random Forest**:
  - Focuses on `smart_9` (Power-on hours).

- **XGBoost**:
  - Prioritizes `smart_198` (Offline uncorrectable sector count), `smart_199` (UltraDMA CRC error count), and `smart_5` (Reallocated sector count).

This indicates that Random Forest emphasizes the age of the HDD, while XGBoost focuses more on physical failure indicators, despite both having almost the same performance.

### Step 8.4: Threshold Adjustment

Adjusts the decision threshold from the default 50% to optimize the F1-Score and recall.

- **Approach**:
  - Test thresholds from 10% to 90% in 1% increments.
  - Select the threshold that maximizes the F1-Score, with a preference for higher recall in case of ties.

- **Results**:
Both models achieved a one-point increase in their F1-Score, with improvements in recall coming at a slight cost to precision.

- **Random Forest**: Showed a slightly higher recall, making it marginally better at identifying failures.
- **XGBoost**: Exhibited a slightly higher precision, indicating it was marginally more accurate in its positive predictions.

Each model thus demonstrated a very slight advantage in either recall (Random Forest) or precision (XGBoost), highlighting their respective strengths.

---

## Model Ensembling

Combining the strengths of both Random Forest and XGBoost models through weighted ensemble stacking to enhance prediction performance.

- **Method**:
  - Assign weights to each model’s predictions.
  - Experiment with various weight combinations and thresholds.
  - Identify the best combinations to maximize F1-Score, precision, and recall.

- **Best Ensemble Combinations**:
  - **Maximize F1-Score**: XGBoost (0.47), Random Forest (0.53), Threshold: 37%
  - **Maximize Precision**: XGBoost (0.34), Random Forest (0.66), Threshold: 90%
  - **Maximize Recall**: XGBoost (0.23), Random Forest (0.77), Threshold: 10%
 
![Ensemble](https://github.com/warc0s/hdd-failure-rate-ml/blob/main/images/Ensemble.png?raw=true)

These combinations provide a balanced approach to handling different aspects of model performance, allowing for a more robust final prediction system.

---

## Implementation and Usage - Tests

The final implementation leverages the ensemble of both models with optimized weights and thresholds to predict HDD failures. The process involves:

1. **Selecting Disks**: Randomly select HDDs from the test set, ensuring at least one has failed for testing purposes.
2. **Applying Ensemble Logic**:
   - **F1-Oriented**: High accuracy in predicting failures.
   - **Precision-Oriented**: Identifies critical failures with high precision.
   - **Recall-Oriented**: Maximizes the detection of actual failures, albeit with some false positives.
3. **Decision Making**:
   - Utilize a series of conditional checks to determine the final prediction based on the ensemble outputs.
   - Provide actionable alerts (prints) based on the combination of model predictions.

**Usage Example**:
![Real Usage Image](https://github.com/warc0s/hdd-failure-rate-ml/blob/main/images/real_usage.png?raw=true)

---

## Conclusions and Future Work

### Key Takeaways

- **Ensemble Approach**: Combining Random Forest and XGBoost leverages their individual strengths, resulting in a more robust prediction system.
- **Threshold Optimization**: Adjusting decision thresholds significantly improves the model’s ability to balance precision and recall.
- **Feature Selection**: Focusing on critical SMART attributes enhances model performance by reducing noise.

### Final Thoughts

This project highlights the effectiveness of machine learning in predicting HDD failures, setting a solid foundation for future, more sophisticated predictive maintenance systems.

As seen in the previous step, the combination of weights and thresholds for the two trained models **performed surprisingly well**—far better than I anticipated.

Initially, I aimed to select a single model, most likely XGBoost, as it's known for reliable performance in similar contexts. My plan was to classify HDDs based on the model's confidence in predicting failure, using labels like 'minor issue, monitor' or 'critical alert, backup recommended.'

However, after plotting the feature importance for each model and observing their different prioritizations, I reconsidered and explored the idea of "blending" the models. I researched various methods for this and discovered **Weighted Ensemble Stacking** with threshold optimization, which is effective for combining model outputs with adjusted importance. The final result, as seen above, achieves a balanced performance, which I find impressive for a first large-scale ML project.

Finally, this project shows promising potential. By focusing on manufacturers rather than specific models and using one-hot encoding, the model could operate with all manufacturer columns set to zero and still perform reliably—placing greater emphasis on SMART attributes. In essence, this is a **universal model** that can generalize across any HDD, making it versatile for broader applications.

---

## Acknowledgements

- **Backblaze**: For providing the comprehensive HDD failure dataset. You can check it [here](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).
- **Groq**: For facilitating access to Llama 3.1 70B for manufacturer extraction.

---

## License

This project is licensed under the Apache2 License.

---

*Thank you for taking the time to explore this project! Feel free to reach out with any questions or suggestions :)*
