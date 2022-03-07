import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

train_labels = []
train_samples = []

"""
Data Creation
For this simple task, we'll be creating our own example data set.
As motivation for this data, let's suppose that an experimental drug 
was tested on individuals ranging from age 13 to 100 in a clinical trial.
The trial had 2100 participants. Half of the participants were under 65 
years old, and the other half was 65 years of age or older.

The trial showed that around 95% of patients 65 or older experienced side
effects from the drug, and around 95% of patients under 65 experienced
no side effects, generally showing that elderly individuals were more 
likely to experience side effects.

Ultimately, we want to build a model to tell us whether or not a patient 
will experience side effects solely based on the patient's age. 
The judgement of the model will be based on the training data.

Note that with the simplicity of the data along with the conclusions 
drawn from it, a neural network may be overkill, but understand this 
is just to first get introduced to working with data for deep learning,
and later, we'll be making use of more advanced data sets.

The block of code below shows how to generate this dummy data.
"""

for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)



for i in train_samples:
    print(i)


for i in train_labels:
    print(i)

train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

for i in scaled_train_samples:
    print(i)



