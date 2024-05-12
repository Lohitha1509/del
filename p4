import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# Generate some random data for two tasks
num_samples = 1000
input_dim = 10
output_dim_task1 = 1
output_dim_task2 = 1

X = np.random.randn(num_samples, input_dim)
Y_task1 = np.random.randn(num_samples, output_dim_task1)
Y_task2 = np.random.randn(num_samples, output_dim_task2)

# Split data into training and validation sets
X_train, X_val, Y_task1_train, Y_task1_val, Y_task2_train, Y_task2_val = train_test_split(
    X, Y_task1, Y_task2, test_size=0.2, random_state=42)

# Define the shared input layer
inputs = Input(shape=(input_dim,))

# Define task-specific hidden layers
shared_hidden = Dense(64, activation='relu')(inputs)

# Define task-specific output layers
output_task1 = Dense(output_dim_task1, name='task1_output')(shared_hidden)
output_task2 = Dense(output_dim_task2, name='task2_output')(shared_hidden)

# Define the model with shared layers
model = Model(inputs=inputs, outputs=[output_task1, output_task2])

# Compile the model
# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error', 'mean_squared_error'])


# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(X_train, [Y_task1_train, Y_task2_train],
                    validation_data=(X_val, [Y_task1_val, Y_task2_val]),
                    epochs=100, batch_size=32, callbacks=[early_stopping])

# Evaluate the model
loss, task1_loss, task2_loss = model.evaluate(X_val, [Y_task1_val, Y_task2_val])

print(f'Overall Loss: {loss}')
print(f'Task 1 Loss: {task1_loss}')
print(f'Task 2 Loss: {task2_loss}')

# Plot training and validation losses for each task
task1_train_loss = history.history['task1_output_mean_squared_error']
task2_train_loss = history.history['task2_output_mean_squared_error']
task1_val_loss = history.history['val_task1_output_mean_squared_error']
task2_val_loss = history.history['val_task2_output_mean_squared_error']

epochs = range(1, len(task1_train_loss) + 1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, task1_train_loss, 'b', label='Task 1 Training loss')
plt.plot(epochs, task1_val_loss, 'r', label='Task 1 Validation loss')
plt.title('Task 1 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.axvline(x=early_stopping.stopped_epoch, color='gray', linestyle='--', label='Early Stopping')

plt.subplot(1, 2, 2)
plt.plot(epochs, task2_train_loss, 'b', label='Task 2 Training loss')
plt.plot(epochs, task2_val_loss, 'r', label='Task 2 Validation loss')
plt.title('Task 2 Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.axvline(x=early_stopping.stopped_epoch, color='gray', linestyle='--', label='Early Stopping')

plt.show()
