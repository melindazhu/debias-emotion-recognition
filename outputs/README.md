# 📂 Output Files  

This directory contains the output files generated, including face embeddings, labels, and fairness evaluation results after applying bias mitigation.  

---

## 📄 File Descriptions  

### 🔹 `raf_embeddings.npy`  
- **Description:** Contains face embeddings generated using the FaceNet model for each image in the RAF-DB dataset.  
- **Format:** Numpy array (`.npy` file), where each entry corresponds to the embedding vector for a specific image.  
- **Usage:** Used as input for training the emotion classification model and performing fairness evaluations.  

### 🔹 `raf_labels.npy`  
- **Description:** Contains race and emotion labels for each image in the RAF-DB dataset.  
- **Format:** Numpy array (`.npy` file), where each entry contains:  
  - **Emotion labels:** Values from 1 to 7, representing the seven basic emotions (happiness, sadness, anger, surprise, fear, disgust, neutral).  
  - **Race labels:** Categories assigned from the FairFace dataset based on facial features.  
- **Usage:** Used to evaluate the performance of the emotion recognition model across different racial groups and analyze fairness.  

### 🔹 `fairness_results.csv`  
- **Description:** Contains fairness evaluation metrics after applying bias mitigation techniques to the emotion classification model.  
- **Format:** CSV file with columns representing different fairness metrics, including:  
  - **Disparate Impact:** Measures fairness across different racial groups.  
  - **Statistical Parity Difference:** Analyzes disparities in prediction rates between groups.  
  - **Confusion Matrix:** Evaluates model accuracy and misclassifications across different races.  
- **Usage:** Helps track the effectiveness of bias mitigation methods and provides insights into fairness improvements.  

---

## 📝 Notes  
- `raf_embeddings.npy` and `raf_labels.npy` serve as input for the emotion classification model and fairness evaluations.  
- `fairness_results.csv` provides evaluation metrics to assess whether the applied bias mitigation techniques improved model fairness.  
