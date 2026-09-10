# Academic Chatbot

An interactive **Academic Chatbot** built with Python, Natural Language Processing (NLP), Machine Learning, and Tkinter. The chatbot is designed to help CSE students with questions related to core academic subjects through a simple desktop interface.

## Project Overview

The chatbot uses a **TF-IDF vectorizer** and **Logistic Regression** classifier to understand user questions and identify the relevant academic intent. It also uses cosine similarity with tag centroids as a fallback for low-confidence predictions.

The application provides a guided conversation where the user enters their name, enrollment number, branch, and preferred subject before asking academic questions.

## Features

- Student information collection
- Subject selection
- Academic question answering
- NLP-based intent classification
- TF-IDF text vectorization
- Logistic Regression classification
- Cosine similarity-based fallback matching
- Tkinter graphical user interface
- Clear Chat functionality
- Change Subject functionality
- Enter key support for sending messages
- Randomized responses for supported intents

## Supported Subjects

The chatbot dataset covers:

- DBMS
- Operating Systems (OS)
- Computer Networks (CN)
- Data Structures and Algorithms (DSA)
- Object-Oriented Programming (OOP)
- Artificial Intelligence (AI)
- Computer Architecture (CA)

## Machine Learning Approach

The chatbot follows this workflow:

```text
User Question
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Logistic Regression
      ↓
Intent Prediction
      ↓
Confidence Check
      ↓
Cosine Similarity Fallback
      ↓
Response Selection
```

### TF-IDF

TF-IDF is used to convert text patterns from the intent dataset into numerical feature vectors.

### Logistic Regression

A Logistic Regression classifier is trained on the TF-IDF features to predict the intent of a user's academic question.

### Cosine Similarity

When the classifier confidence is low, cosine similarity is used with tag centroids to find a closer matching intent.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Tkinter | Graphical user interface |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and NLP |
| TF-IDF | Text feature extraction |
| Logistic Regression | Intent classification |
| Cosine Similarity | Fuzzy intent matching |
| JSON | Intent and response dataset |
| Pickle | Model and processed data storage |

## Project Structure

```text
academic-chatbot/
│
├── README.md
├── chatbot.py
├── chatbot_gui.py
├── intents.json
├── chatbot_model.pkl
├── vectorizer.pkl
├── tag_responses.pkl
├── tag_centroids.pkl
└── requirements.txt
```

## File Description

- **chatbot.py** — trains the NLP model and generates the required model/data files.
- **chatbot_gui.py** — runs the Tkinter-based chatbot interface.
- **intents.json** — contains academic intents, question patterns, and responses.
- **chatbot_model.pkl** — trained Logistic Regression model.
- **vectorizer.pkl** — trained TF-IDF vectorizer.
- **tag_responses.pkl** — stored intent-response mappings.
- **tag_centroids.pkl** — stored intent centroid vectors used for similarity matching.
- **requirements.txt** — Python dependencies required to run the project.

## Installation

Clone the repository:

```bash
git clone https://github.com/rajprashant2064/academic-chatbot.git
cd academic-chatbot
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Chatbot

The repository includes the trained model files, so you can start the GUI directly:

```bash
python chatbot_gui.py
```

The chatbot will open in a Tkinter window.

## Retrain the Model

If you modify `intents.json`, retrain the model before running the GUI:

```bash
python chatbot.py
```

This generates:

```text
chatbot_model.pkl
vectorizer.pkl
tag_responses.pkl
tag_centroids.pkl
```

Then run:

```bash
python chatbot_gui.py
```

## User Flow

```text
Enter Name
    ↓
Enter Enrollment Number
    ↓
Enter Branch
    ↓
Select Subject
    ↓
Ask Academic Questions
    ↓
Receive Chatbot Responses
```

## Skills Demonstrated

- Python Programming
- Natural Language Processing
- Machine Learning
- Text Classification
- TF-IDF Feature Engineering
- Logistic Regression
- Cosine Similarity
- GUI Development with Tkinter
- JSON Data Handling
- Model Serialization with Pickle

## Author

**Prashant Raj**

B.Tech Computer Science & Engineering

- LinkedIn: https://www.linkedin.com/in/prashantraj2064
- GitHub: https://github.com/rajprashant2064
