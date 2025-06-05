import numpy as np
from proj import preprocess,preprocess_single

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

    def backward(self, x, y, output, lr=0.01):
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

    def train(self, X, Y, epochs=100):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.backward(X, Y, output)
            pred = (output >= 0.5).astype(int)
            acc = np.mean(pred == Y)
            print(f"Epoch {epoch+1:03} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
        print(f"\nTraining complete. Final accuracy: {acc:.4f}")

        # Final predictions
        # print("\nFinal Predictions:")
        # for i in range(len(X)):
        #     pred = self.forward(X[i].reshape(1, -1))
        #     label = 1 if pred >= 0.5 else 0
        #     print(f"Sample {i + 1}: Prediction = {label}, True = {int(Y[i][0])}")
    def predict_sample(self, x_sample):
        output = self.forward(x_sample)
        return 1 if output >= 0.5 else 0

if __name__ == "__main__":
    # Load and preprocess your dataset
    X, Y, vocab_size, vocab = preprocess("tasks.csv")

    # Initialize and train the model
    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=16, output_size=1)
    nn.train(X, Y, epochs=350)

    # Predict a custom task
    sample = preprocess_single("Call exercise", "Personal", "Low", "2025-06-28", vocab)
    predicted_label = nn.predict_sample(sample)
    print(f"\nCustom task prediction: {predicted_label}")

