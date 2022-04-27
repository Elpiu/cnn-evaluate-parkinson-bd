import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

from fristModelDataPreparation import *
 
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])


model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x=scaled_train_samples, y=train_labels,
            validation_split=0.1, batch_size=10, epochs=30,
            shuffle=True, verbose=2)

#validation_split
v = "/"*10+"Testing"+"/"*10
print(v*10)
from testData import *
predictions = model.predict(
      x=scaled_test_samples
    , batch_size=10
    , verbose=0
)  


for i in predictions:
    print(i)










