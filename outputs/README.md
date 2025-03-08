# 📂 Output Files  

This directory contains the output files, including face embeddings, labels, and fairness evaluation results after bias mitigation.  

---

## 📄 File Descriptions  

### 🔹 `raf_embeddings.npy`  
- **Description:** Face embeddings generated using the FaceNet model for images in the RAF-DB dataset.  
- **Format:** Numpy array (`.npy`), where each entry is an embedding vector for a specific image.  
- **Usage:** Used for training the emotion classification model and fairness evaluations.  

### 🔹 `raf_labels.npy`  
- **Description:** Race and emotion labels for each image in the RAF-DB dataset.  
- **Format:** Numpy array (`.npy`), where each entry contains:  
  - **Emotion labels:** Values from 1 to 7, representing the seven basic emotions (happiness, sadness, anger, surprise, fear, disgust, neutral).  
  - **Race labels:** Categories assigned using the FairFace dataset.  
- **Usage:** Evaluates emotion recognition model performance across racial groups and assesses fairness.  

### 🔹 `fairness_results.csv`  
- **Description:** Fairness evaluation metrics after applying bias mitigation to the emotion classification model.  
- **Format:** CSV file with columns for various fairness metrics, including:  
  - **Disparate Impact:** Measures fairness across racial groups.  
  - **Statistical Parity Difference:** Analyzes disparities in prediction rates.  
  - **Confusion Matrix:** Assesses accuracy and misclassifications by race.  
- **Usage:** Tracks bias mitigation effectiveness and fairness improvements.  

---

## 📝 Notes  
- `raf_embeddings.npy` and `raf_labels.npy` are inputs for the emotion classification model and fairness evaluations.  
- `fairness_results.csv` provides key metrics to determine the impact of bias mitigation.  
