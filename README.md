# 🧠 Streamlit Digit Classifier

A simple Streamlit web app that recognizes hand-drawn digits (0–9) using a PyTorch-trained MNIST model.

## ✨ Demo

Draw a digit in the canvas, click **Submit Drawing**, and the model will predict what number you drew — along with its confidence and class probabilities.

![screenshot](./screenshot.png) <!-- Optional: add a screenshot of your app UI -->



## 🗂️ Project Structure
```
streamlit-digit-classifier-app/
│
├── app.py # Streamlit app UI and inference logic
├── model.py # PyTorch DigitClassifier model definition
├── mnist_model.pth # Trained model weights
├── requirements.txt # Python dependencies
├── .env # Environment variables for database connection
├── Dockerfile # Dockerfile for the Streamlit web app
├── docker-compose.yml # Docker Compose file to run the app and database
└── README.md # Project documentation
```


## 🛠️ Setup Instructions

### Option 1. Run with Docker (Recommended)

This is the easiest way to get the app running, as it handles all dependencies and the database setup automatically.

#### 1. Clone the Repository

```
git clone https://github.com/alicespylgass/streamlit-digit-classifier-app.git
cd streamlit-digit-classifier-app
```

#### 2. Set up Your Environment

Create a `.env` file in the root directory of your project to store your database credentials.

```
DB_USER=your_postgres_user
DB_PASSWORD=your_password
DB_NAME=digit_classifier
```

Replace the values with your actual PostgreSQL credentials.

Note: Do not commit your .env file to version control.


#### 3. Run the App

Ensure you have **Docker Desktop** installed and running on your machine. From the project's root directory, run the following command to build and start the containers.

```
docker-compose up --build
```

The app will automatically create the `predictions` table in the database. Once the containers are running, navigate to `http://localhost:8501` in your browser.


### Option 2. Run locally

Use this option if you prefer to manage the dependencies and database directly on your local machine.

#### 1. Clone the Repository

```
git clone https://github.com/alicespylgass/streamlit-digit-classifier-app.git
cd streamlit-digit-classifier-app
```

#### 2. Create a Virtual Environment (Recommended)

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Set up PostgreSQL and `.env`
- Start your local PostgreSQL server.

- Create a .env file in the root directory of your project with your database credentials.

```
DATABASE_URL=postgresql://your_postgres_user:your_password@localhost:5432/digit_classifier
```
Replace the values with your actual PostgreSQL credentials.

Note: Your app will automatically create the `predictions` table when you run it for the first time.


### 5. Run the App
```
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

### ✍️ Providing Model Feedback
To help improve the model, you can provide feedback on incorrect predictions.

After the model makes a prediction, a feedback form will appear.

If the prediction is correct, you can confirm it.

If the prediction is wrong, select the correct digit from the dropdown menu and submit the feedback.

This feedback will be stored in the `predictions` table, and the `is_correct` and `feedback_given` columns will be updated. This data can be used in the future to retrain and improve the model.


### 🧠 Model Details

Architecture: Simple CNN with ReLU activations and linear layers

Dataset: MNIST handwritten digits

Training: 5 epochs on 60,000 training samples

Model is saved as mnist_model.pth. You can retrain your own model using the same architecture in model.py.

### 🔁 Features

- Interactive drawing canvas
- Dynamic preprocessing pipeline: crop → resize → pad → normalize
- Predictions with confidence and class probabilities
- Feedback form for correcting predictions (for future retraining)
- Reset button to clear canvas

### 🧪 TODO / Improvements
- [ ] Improve prediction accuracy (train longer or augment data)
- [ ] Deploy to the web (e.g. Streamlit Cloud or Hugging Face Spaces)


### 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [streamlit-drawable-canvas](https://github.com/andfanilo/streamlit-drawable-canvas)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)