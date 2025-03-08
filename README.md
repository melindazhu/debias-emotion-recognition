# **Detecting and Mitigating Bias in Face Emotion Recognition Models**  

### **👥 Students:** Basant Khalil & Melinda Zhu  

---

## **📌 Project Overview**  

Facial emotion recognition (FER) models are widely used in fields such as human-computer interaction, security, and psychology. However, these models often exhibit **racial and ethnic biases** due to imbalanced datasets, leading to **unfair predictions** across different demographic groups.  

Our project aims to:  
✔ Train a **baseline emotion classification model** using real-world datasets.  
✔ **Evaluate bias** in emotion recognition across racial and ethnic groups.  
✔ Implement **bias mitigation strategies** to enhance fairness in FER models.  

By integrating **RAF-DB** (emotion-labeled dataset) with **FairFace** (race-labeled dataset), we create a more balanced dataset for **bias-aware facial emotion classification**.  

---

## **📖 Background**  

Studies show significant racial and gender biases in commercial **FER models**. For example:  

🔹 **Buolamwini & Gebru (2018)**: Found racial biases in face analysis systems.  
🔹 **Zhang et al. (2018)**: Proposed **adversarial debiasing** to mitigate bias.  

### **🧑‍💻 Why does this bias exist?**  
- **Imbalanced datasets** → Certain racial groups are underrepresented.  
- **Feature learning issues** → Models may encode **racial attributes** instead of emotion-based patterns.  
- **Historical biases in AI** → Pretrained models often inherit bias from their training data.  

Our project plans to **quantify and mitigate** these biases using fairness-aware learning techniques.  

---

## **🛠 Methodology**  

### **1️⃣ Data Processing**  
✔ **Assign race labels** to the RAF-DB dataset using FairFace.  
✔ **Extract face embeddings** using **FaceNet** for race-label matching.  

### **2️⃣ Baseline Model Training**  
✔ Train a **ResNet**/**MobileNet** model on **RAF-DB**.  
✔ Evaluate bias across racial groups using:  
   - **Per-class accuracy**  
   - **False positive/negative rates**  
   - **Disparate impact & statistical parity difference**  

### **3️⃣ Bias Mitigation Techniques**  
✔ Implement **fairness-aware adversarial networks** to remove race-sensitive features.  
✔ Retrain the model and **recompute fairness metrics**.  

---

## **📂 Directory Structure**  

```bash
Face_Emotion_Bias_Project/
├── data/
│   ├── RAF_DB/          # RAF-DB dataset (emotion labels)
│   ├── FairFace/        # FairFace dataset (race/ethnicity labels)
│   └── README.md        # Dataset details
├── outputs/  
│   ├── raf_embeddings.npy      # Processed face embeddings  
│   ├── raf_labels.npy          # Processed race & emotion labels  
│   ├── fairness_results.csv    # Bias evaluation results  
│   └── README.md               # Output explanations  
├── code/  
│   ├── data_processing.py  # Preprocess datasets & extract embeddings  
│   ├── baseline_model.py   # Train baseline emotion model  
│   ├── bias_mitigation.py  # Implement bias mitigation techniques  
│   └── README.md           # Code structure explanation  
├── requirements.txt   # Required Python packages  
├── README.md          # Project overview & setup instructions  
└── .gitignore         # Ignore large files (e.g., datasets)
```

---
# 🚀 Setup Instructions  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/Face_Emotion_Bias_Project.git
cd Face_Emotion_Bias_Project
```

### 2️⃣ Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
```

