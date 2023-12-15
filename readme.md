# Email Classifier Project

## Introduction

This project is an Email Classifier, developed by Sufian. It's designed to classify emails as either 'Spam' or 'Ham' (non-spam) using a web interface.

## Features

- **Email Classification**: Allows users to input an email message and classify it as Spam or Ham.
- **User Feedback**: Users can provide feedback on the accuracy of the classification.
- **Dual Classification Models**: Employs both TfidfVectorizer and CountVectorizer for email classification.

## How It Works

1. **Input Email**: Users input an email message in the provided text area.
2. **Classification**: Upon clicking "Classify Email", the message is sent to the backend server for classification.
3. **Result Display**: The classification result is shown, indicating whether the email is Spam or Ham as per each model.
4. **Feedback Submission**: Users have the option to submit feedback on the accuracy of the prediction.
5. **Feedback Retraining**: The Model is retrained after each Feedback submission.

## Installation

### Prerequisites

- Python 3.x
- Flask

### Setup Instructions

1. Clone the repository.
2. Install the required dependencies, including Flask.
3. Run `app.py` to initiate the Flask server.

## Usage

1. Open `index.html` in a web browser.
2. Enter the email text to be classified.
3. Click "Classify Email" to view predictions.
4. Optionally, provide feedback on the prediction's accuracy.

## Technologies

- **Frontend**: HTML, JavaScript
- **Backend**: Python (Flask)
- **ML Models**: TfidfVectorizer, CountVectorizer

**Note**: For full functionality, ensure all files (`app.py`, `index.html`, `styles.css`) are correctly placed in the project directory and the Flask server is active.
