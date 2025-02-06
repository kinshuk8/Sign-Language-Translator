# Real-Time Sign Language Translator

## 📌 Project Overview
This project is a **real-time sign language translator** using a deep learning model trained on sign language gestures. The system captures live hand gestures through a webcam, processes them using a Convolutional Neural Network (CNN), and translates them into corresponding alphabets.

## 📂 Dataset
The dataset used for training consists of images of sign language alphabets, with each alphabet stored in its own directory.

📌 **Dataset Source:** [link](https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets/code)

## ⚙️ Installation & Setup
Follow these steps to set up and run the project:

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/kinshuk8/Sign-Language-Translator.git
cd sign_language_translator
```

### 2️⃣ **Create & Activate a Virtual Environment**
```sh
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 3️⃣ **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4️⃣ **Train the Model (Optional)**
If you want to train the model from scratch, run:
```sh
python scripts/train_model.py
```

### 5️⃣ **Run the Real-Time Translator**
```sh
python scripts/real_time_translator.py
```

## 🖥️ Project Structure
```
├── dataset/                # Sign language images
├── models/                 # Trained models
├── scripts/
│   ├── train_model.py      # Model training script
│   ├── real_time_translator.py  # Real-time translation
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
```

## 🤖 Technologies Used
- **Python**
- **TensorFlow / Keras** (for deep learning)
- **OpenCV** (for real-time image processing)
- **NumPy, Matplotlib** (for data handling & visualization)

## 🔥 Future Improvements
- Improve model accuracy with more training data.
- Implement word-level recognition instead of individual alphabets.
- Add support for different sign languages.

## 📬 Contributing
Feel free to fork this repository, submit pull requests, or report issues!

