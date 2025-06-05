import numpy as np
from proj import preprocess_with_category,preprocess, preprocess_single  # use the earlier function you wrote
from sklearn.model_selection import train_test_split  # only for splitting — can replace if needed


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, output, lr=0.005):
        error = y - output
        d_output = error * self.sigmoid_deriv(output)

        error_hidden = d_output.dot(self.w2.T)
        d_hidden = error_hidden * self.sigmoid_deriv(self.a1)

        self.w2 += self.a1.T.dot(d_output) * lr
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * lr
        self.w1 += x.T.dot(d_hidden) * lr
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

        loss = np.mean(np.square(error))
        return loss

    def train(self, X_train, Y_train, X_test, Y_test, epochs=300, task_name=""):
        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = self.backward(X_train, Y_train, output)

            # Predict for training
            pred_train = (output >= 0.5).astype(int) if Y_train.shape[1] == 1 else (output == output.max(axis=1, keepdims=True)).astype(int)
            acc_train = np.mean(pred_train == Y_train)

            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"[{task_name}] Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Train Accuracy: {acc_train:.4f}")

        # After training, evaluate on test set
        output_test = self.forward(X_test)
        pred_test = (output_test >= 0.5).astype(int) if Y_test.shape[1] == 1 else (output_test == output_test.max(axis=1, keepdims=True)).astype(int)
        acc_test = np.mean(pred_test == Y_test)

        print(f"[{task_name}] Final Train Accuracy: {acc_train:.4f} | Final Test Accuracy: {acc_test:.4f}")
        return acc_train, acc_test  


    def predict_sample(self, x_sample):
        output = self.forward(x_sample)
        if output.shape[1] == 1:
            return int(output >= 0.5)
        else:
            return int(np.argmax(output))

    def save_weights(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)

    def load_weights(self, filename):
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']


if __name__ == "__main__":
    # Load and preprocess
    X, Y_priority, Y_category, vocab, category_map = preprocess_with_category("balanced_tasks.csv")
 
    # Normalize features
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("category_map.pkl", "wb") as f:
        pickle.dump(category_map, f)
    print("vocab.pkl and category_map.pkl saved.")
    # Split dataset
    X_train, X_test, yp_train, yp_test = train_test_split(X, Y_priority, test_size=0.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(X, Y_category, test_size=0.2, random_state=42)

    input_size = X.shape[1]
    hidden_size = 16
    
    
    # --- Category Model (Multi-class Classifier) ---
    category_model = NeuralNetwork(input_size, hidden_size, Y_category.shape[1])
    acc_category = category_model.train(X_train, yc_train,X_test, yc_test, epochs=400, task_name="Category")
    category_model.save_weights("category_model.npz")
    print("Category model saved.")                      

      # --- Priority Model (Binary Classifier) ---
    priority_model = NeuralNetwork(input_size=len(vocab), hidden_size=16, output_size=3)
    acc_priority = priority_model.train(X_train, yp_train, X_test, yp_test, epochs=450, task_name="Priority")
    priority_model.save_weights("priority_model.npz")
    print("Priority model saved.")