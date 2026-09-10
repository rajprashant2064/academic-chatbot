import tkinter as tk
from tkinter import scrolledtext
import pickle
import random

def identity_tokenizer(text):
    return text

def identity_preprocessor(text):
    return text

with open("chatbot_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("tag_responses.pkl", "rb") as f:
    tag_responses = pickle.load(f)

with open("tag_centroids.pkl", "rb") as f:
    tag_centroids = pickle.load(f)

user_info = {"name": "", "enroll": "", "branch": "", "subject": ""}
stage = 0

def get_bot_response(user_input):
    global stage

    user_input = user_input.strip().lower()

    if stage == 0:
        user_info["name"] = user_input
        stage += 1
        return "Please enter your enrollment number."

    elif stage == 1:
        user_info["enroll"] = user_input
        stage += 1
        return "Please enter your Branch (e.g., CSE, IT, ECE)."

    elif stage == 2:
        user_info["branch"] = user_input
        stage += 1
        return "For which subject do you need help? (DBMS / OS / CN / DSA / OOP / AI / CA)"

    elif stage == 3:
        user_info["subject"] = user_input.upper()
        stage += 1
        return f"Thanks {user_info['name'].title()} ({user_info['enroll']}). You selected {user_info['subject']} — now ask any question about this subject."

    else:
        subj = user_info["subject"].lower()
        vec = vectorizer.transform([user_input])
        tag = model.predict(vec)[0]

        if tag not in tag_responses:
            best_match = None
            best_score = 0
            for t, centroid in tag_centroids.items():
                sim = (vec @ centroid.T).toarray().flatten()[0]
                if sim > best_score:
                    best_match, best_score = t, sim
            tag = best_match if best_match else "unknown"

        responses = tag_responses.get(tag, [f"{user_info['subject'].upper()} info."])
        return random.choice(responses)

def send_message(event=None):
    user_input = entry_field.get().strip()
    if not user_input:
        return

    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"You: {user_input}\n")

    response = get_bot_response(user_input)
    chat_window.insert(tk.END, f"Bot: {response}\n\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.see(tk.END)
    entry_field.delete(0, tk.END)

def clear_chat():
    global stage, user_info
    chat_window.config(state=tk.NORMAL)
    chat_window.delete(1.0, tk.END)
    chat_window.config(state=tk.DISABLED)
    stage = 0
    user_info = {"name": "", "enroll": "", "branch": "", "subject": ""}
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Bot: Hi\nPlease enter your name.\n")
    chat_window.config(state=tk.DISABLED)

def change_subject():
    global stage
    stage = 3
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "Bot: You can now change your subject. (DBMS / OS / CN / DSA / OOP / AI / CA)\n")
    chat_window.config(state=tk.DISABLED)
    chat_window.see(tk.END)

root = tk.Tk()
root.title("Academic Chatbot - Friendly (CSE)")
root.geometry("800x600")

frame = tk.Frame(root)
frame.pack(pady=5)

clear_button = tk.Button(frame, text="Clear Chat", command=clear_chat)
clear_button.grid(row=0, column=0, padx=5)

subject_button = tk.Button(frame, text="Change Subject", command=change_subject)
subject_button.grid(row=0, column=1, padx=5)

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry_field = tk.Entry(root, width=80)
entry_field.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
entry_field.bind("<Return>", send_message)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.RIGHT, padx=10, pady=10)

chat_window.config(state=tk.NORMAL)
chat_window.insert(tk.END, "Bot: Hi\nPlease enter your name.\n")
chat_window.config(state=tk.DISABLED)

root.mainloop()