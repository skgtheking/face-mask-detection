# Face Mask Detection System 😷

This project implements a **real-time face mask detection system** using computer vision and machine learning. Built for practical deployment during the COVID-19 pandemic, it classifies whether a person is wearing a mask using live webcam input.

---

## 🚀 Features
- Real-time detection using OpenCV
- High-accuracy classification with **SVM**
- Lightweight and optimized for standard hardware
- Clean, scalable code structure
- Optional CNN version for advanced use cases

---

## 🧠 Tech Stack

| Area            | Tools Used                             |
|-----------------|-----------------------------------------|
| Language        | Python                                  |
| Libraries       | OpenCV, scikit-learn, joblib, NumPy     |
| Model           | Support Vector Machine (SVM)            |
| Detection       | Haar Cascade Classifier                 |
| Deployment      | Local with webcam (via OpenCV)          |

---

## 🗂 Project Structure

```
Face_Mask_Detection_Project/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── preprocessed_data/         # Not pushed to GitHub (too large)
├── model/
│   └── svm_model.joblib
├── haarcascade_frontalface.xml
├── mask_detector.py           # Main script
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/skgtheking/face-mask-detection.git
cd face-mask-detection

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the detector
python mask_detector.py
```

---

## 🧪 Model Performance

- **Accuracy:** 90% on test set
- Handles masked vs unmasked faces reliably in real-time
- Misclassifications mainly under low light or occlusion

---

## 📌 Future Improvements

- Switch to lightweight CNN (e.g., MobileNet) for higher accuracy
- Integrate mask type classification (e.g., cloth vs N95)
- Deploy via Flask web app or Streamlit

---

## 🧑‍💻 Author

**Shubham Gupta**  
Computer Science Graduate, University of Idaho  
📫 [LinkedIn](https://www.linkedin.com/in/shubham-gupta-891a831b2) • [GitHub](https://github.com/skgtheking)
