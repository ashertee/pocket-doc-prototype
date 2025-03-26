import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#Todo: Add unit test files


# Create a simple mock model for demonstration purposes
def create_mock_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(100,)),  # Assuming 100 input features
        Dense(4, activation='softmax')  # Assuming 4 possible diagnoses
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Generate some mock data for training
X_train = np.random.random((1000, 100))
y_train = np.random.randint(4, size=(1000, 1))
y_train = tf.keras.utils.to_categorical(y_train, 4)

# Create and train the mock model
model = create_mock_model()
model.fit(X_train, y_train, epochs=5)

# Save the model to 'model.h5'
model.save('model.h5')
