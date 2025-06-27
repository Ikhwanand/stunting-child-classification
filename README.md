# 👶 Stunting Child Classification App

A Streamlit web application for classifying and analyzing child stunting/nutritional status with machine learning.

## 📌 Features

- **Dashboard**: Overview metrics and visualizations
- **Data Analysis**: Exploratory data analysis with interactive plots
- **Prediction**: ML model to predict child's nutritional status
- **Visualization**: Interactive data visualizations
- **Model Info**: Details about the trained Random Forest model

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit (web framework)
- Scikit-learn (machine learning)
- Plotly (visualizations)
- Pandas (data processing)
- Joblib (model serialization)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/ikhwananda/stunting-child-classification.git
cd stunting-child-classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the App

```bash
streamlit run main.py
```

The app will launch in your default browser at `http://localhost:8501`

## 📂 Project Structure

```
stunting-child-classification/
├── data/                   # Dataset folder
│   └── data_balita.csv     # Child nutrition data
├── models/                 # Trained models
│   ├── le_gender.pkl       # Gender label encoder
│   ├── le_status.pkl       # Status label encoder  
│   ├── rf_models.joblib    # Random Forest model
│   └── scaler.pkl          # Feature scaler
├── notebooks/              # Jupyter notebooks
│   └── nb-1.ipynb          # Data exploration notebook
├── agent.py                # Recommendation agent
├── main.py                 # Main Streamlit app
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore rules
```

## 🤖 Model Details

The app uses a **Random Forest Classifier** with:
- Feature importance analysis
- Probability distribution visualization
- Input preprocessing (scaling and encoding)

Key features used:
- Age (months)
- Gender
- Height (cm)

## 📊 Data

Sample data includes:
- 1000 records of child nutrition data
- Features: Age, Gender, Height
- Target: Nutritional Status (normal, stunted, severely stunted, tall)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌐 Data source

This dataset from kaggle with [Link Dataset](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows/data?select=data_balita.csv)

## ⚠️ Disclaimer

This application is for educational purposes only and does not replace professional medical consultation.