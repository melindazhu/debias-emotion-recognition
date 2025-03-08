# Detecting and Mitigating Bias in Face Emotion Recognition Models

## 📌 Project Overview
This project aims to detect and mitigate bias in facial emotion recognition (FER) models by leveraging emotion and racial labels from the **RAF-DB** and **FairFace** datasets. The primary goals are:
- Evaluate the baseline performance of an emotion classification model across different racial groups.
- Identify potential biases in model predictions.
- Implement bias mitigation techniques to improve fairness.

---

## 📂 Folder Structure
```
/project-root
│
├── data/
│   ├── RAF_DB/                # RAF-DB dataset with train and test splits
│   │   ├── train/             # Emotion-labeled images in subdirectories 1-7
│   │   └── test/              # Same structure as train
│   │
│   ├── FairFace/              # FairFace dataset with race annotations
│   │   ├── train/             # Images with race, age, and ethnicity annotations
│   │   └── test/              # Same structure as train
│
├── src/
│   ├── data_processing.py     # Data processing for RAF-DB & FairFace
│   ├── baseline_model.py      # Train & evaluate the baseline model
│   └── bias_mitigation.py     # Implement and evaluate bias mitigation
│
├── raf_embeddings.npy         # Preprocessed face embeddings for RAF-DB
├── raf_labels.npy             # Race and emotion labels for RAF-DB images
└── README.md                  # This README file
```

---

## 📊 Data Description
### 🔹 **RAF-DB Dataset** (Stored in `data/RAF_DB/`)
- Contains images labeled with **seven emotions**: happiness, sadness, anger, surprise, fear, disgust, and neutral.
- Organized into `train/` and `test/` folders, where each subdirectory (1-7) represents an emotion category.
- **No racial labels available**, requiring integration with FairFace for fairness evaluation.

### 🔹 **FairFace Dataset** (Stored in `data/FairFace/`)
- Provides **race, age, and ethnicity annotations** for facial images.
- Used to **assign race labels** to RAF-DB images for fairness assessment.

---

## 📥 How to Obtain the Data
1. **Download RAF-DB:** Obtain from the [official RAF-DB website](https://www.whdeng.cn/RAF/model1.html) and extract into `data/RAF_DB/`.
2. **Download FairFace:** Get it from the [official FairFace GitHub](https://github.com/joojs/fairface) and extract into `data/FairFace/`.

---

## 🚀 Running the Code
### 1️⃣ Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-username/face-emotion-bias.git
cd face-emotion-bias
pip install -r requirements.txt
```

### 2️⃣ Process the Data
Extract face embeddings and assign race labels:
```bash
python src/data_processing.py
```

### 3️⃣ Train & Evaluate the Baseline Model
```bash
python src/baseline_model.py
```

### 4️⃣ Apply Bias Mitigation Techniques
```bash
python src/bias_mitigation.py
```

---

## 📝 Notes
- Ensure RAF-DB and FairFace datasets are **downloaded and placed correctly** before running the scripts.
- **Bias Evaluation:** The model is assessed for bias across racial groups using statistical fairness metrics.
- **Bias Mitigation:** Various fairness-enhancing techniques are tested to reduce disparities in emotion recognition.

---

## 📚 References
- **RAF-DB Dataset:** [Official Page](https://www.whdeng.cn/RAF/model1.html)
- **FairFace Dataset:** [GitHub](https://github.com/joojs/fairface)
- **Fairness Metrics in ML:** [Google AI Blog](https://ai.googleblog.com/2020/05/fairness-in-machine-learning.html)
