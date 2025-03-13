import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='relu', input_shape=(2,)),  # Hidden layer with 3 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron 
])

model.compile(optimizer='adam', loss='binary_crossentropy')

X = np.array([[0,0], [0,1], [1,0], [1,1]])  # Inputs
Y = np.array([[0], [1], [1], [0]])  # Expected Outputs

history = model.fit(X, Y, epochs=10000, verbose=2)  


predictions = model.predict(X)
print("Predictions:")

for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted Output: {predictions[i][0]:.4f}")

plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()