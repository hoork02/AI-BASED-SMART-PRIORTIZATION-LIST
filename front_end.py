import tkinter as tk
from tkinter import ttk, messagebox
import csv
import os
import numpy as np
import pickle
from final2 import NeuralNetwork  # replace 'your_model_file' with actual filename

# Load vocab and category map saved earlier
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("category_map.pkl", "rb") as f:
    category_map = pickle.load(f)

# Import your NeuralNetwork class and load weights

# Load models
priority_model = NeuralNetwork(len(vocab), 16, 1)
priority_model.load_weights("priority_model.npz")

category_model = NeuralNetwork(len(vocab), 16, len(category_map))
category_model.load_weights("category_model.npz")

def preprocess_single_task(task):
    x = np.zeros(len(vocab))
    for word in task.lower().split():
        if word in vocab:
            x[vocab[word]] += 1
    return x.reshape(1, -1)

def predict_priority(task):
    x = preprocess_single_task(task)
    return "High" if priority_model.predict_sample(x) == 1 else "Low"

def predict_category(task):
    x = preprocess_single_task(task)
    cat_idx = category_model.predict_sample(x)
    inv_map = {v:k for k,v in category_map.items()}
    return inv_map.get(cat_idx, "Unknown")

# CSV filenames
USER_CSV = "users_new.csv"
TASK_CSV = "tasks_new.csv"

# Ensure files exist
if not os.path.exists(USER_CSV):
    with open(USER_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["username", "password"])

if not os.path.exists(TASK_CSV):
    with open(TASK_CSV, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["username", "task", "due_date", "priority", "category"])

# --- GUI APP ---
class TaskApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Task Manager")
        self.geometry("800x600")
        self.resizable(False, False)

        self.logged_in_user = None
        self.frames = {}

        for F in (LoginSignupPage, TaskPage):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.place(relwidth=1, relheight=1)

        self.show_frame("LoginSignupPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class LoginSignupPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Background & style
        self.configure(bg="#f7c59f")

        tk.Label(self, text="Welcome! Please Login or Signup", font=("Arial", 18, "bold"), bg="#f7c59f").pack(pady=20)

        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        tk.Label(self, text="Username:", bg="#f7c59f").pack()
        tk.Entry(self, textvariable=self.username_var).pack(pady=5)

        tk.Label(self, text="Password:", bg="#f7c59f").pack()
        tk.Entry(self, textvariable=self.password_var, show="*").pack(pady=5)

        btn_frame = tk.Frame(self, bg="#f7c59f")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Login", command=self.login).grid(row=0, column=0, padx=10)
        tk.Button(btn_frame, text="Signup", command=self.signup).grid(row=0, column=1, padx=10)

    def login(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password.")
            return
        with open(USER_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username and row['password'] == password:
                    self.controller.logged_in_user = username
                    messagebox.showinfo("Success", f"Welcome back, {username}!")
                    self.controller.show_frame("TaskPage")
                    return
            messagebox.showerror("Error", "Invalid username or password.")

    def signup(self):
        username = self.username_var.get().strip()
        password = self.password_var.get().strip()
        if not username or not password:
            messagebox.showerror("Error", "Please enter username and password.")
            return

        with open(USER_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == username:
                    messagebox.showerror("Error", "Username already exists!")
                    return

        with open(USER_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([username, password])
        messagebox.showinfo("Success", "Account created! Please login.")
        self.username_var.set("")
        self.password_var.set("")

class TaskPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="#a3d2ca")

        # Task Input Form
        form_frame = tk.Frame(self, bg="#a3d2ca")
        form_frame.pack(pady=10, fill='x')

        tk.Label(form_frame, text="Task:", bg="#a3d2ca").grid(row=0, column=0, padx=5, sticky='e')
        self.task_var = tk.StringVar()
        tk.Entry(form_frame, textvariable=self.task_var, width=50).grid(row=0, column=1, padx=5)

        tk.Label(form_frame, text="Due Date (YYYY-MM-DD):", bg="#a3d2ca").grid(row=1, column=0, padx=5, sticky='e')
        self.due_var = tk.StringVar()
        tk.Entry(form_frame, textvariable=self.due_var, width=20).grid(row=1, column=1, padx=5, sticky='w')

        tk.Button(form_frame, text="Add Task", command=self.add_task).grid(row=2, column=1, pady=10, sticky='w')

        # Task Display Table
        self.tree = ttk.Treeview(self, columns=("Task", "Due Date", "Priority", "Category"), show='headings')
        self.tree.heading("Task", text="Task")
        self.tree.heading("Due Date", text="Due Date")
        self.tree.heading("Priority", text="Priority")
        self.tree.heading("Category", text="Category")
        self.tree.pack(expand=True, fill='both', pady=10, padx=10)

        # Logout Button
        tk.Button(self, text="Logout", command=self.logout, bg="#e76f51", fg="white").pack(pady=5)

        self.load_tasks()

    def add_task(self):
        task = self.task_var.get().strip()
        due = self.due_var.get().strip()

        if not task or not due:
            messagebox.showerror("Error", "Please enter task and due date.")
            return

        # Predict priority & category
        priority = predict_priority(task)
        category = predict_category(task)

        username = self.controller.logged_in_user

        # Save task in CSV
        with open(TASK_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([username, task, due, priority, category])

        messagebox.showinfo("Task Added", f"Task added with priority {priority} and category {category}")

        self.task_var.set("")
        self.due_var.set("")

        self.load_tasks()

    def load_tasks(self):
        # Clear table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Load user's tasks
        tasks = []
        with open(TASK_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['username'] == self.controller.logged_in_user:
                    tasks.append(row)

        # Sort by priority: High > Medium > Low
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        tasks.sort(key=lambda x: priority_order.get(x['priority'], 3))

        # Insert into treeview
        for t in tasks:
            self.tree.insert("", "end", values=(t['task'], t['due_date'], t['priority'], t['category']))

    def logout(self):
        self.controller.logged_in_user = None
        self.controller.show_frame("LoginSignupPage")

if __name__ == "__main__":
    app = TaskApp()
    app.mainloop()
