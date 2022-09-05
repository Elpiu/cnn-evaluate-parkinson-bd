import numpy as np
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

#Percorso cartella per il testing
pathTest = "C:\\Users\\elpid\\Desktop\\DataSetNew\\converted\\d\\seg_test"
#Percorso cartella per il training
pathTrain = "C:\\Users\\elpid\\Desktop\\DataSetNew\\converted\\d\\seg_train"


#Nomi delle 4 labels
class_names = ['1', '2', '3', '4']
#Enumerazione chiave valore delle classi
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
#Numero delle classi
nb_classes = len(class_names)



#Carica le immagini dalle due cartelle indicate come parametri
def load_data(seg_trainPath, seg_testPath):

    datasets = [seg_trainPath, seg_testPath]
    output = []

    # Iterate through training and test sets
    for dataset in datasets:

        images = []
        labels = []

        print("Loading {}".format(dataset))

        # Iterate through each folder corresponding to a category
        for folder in os.listdir(dataset):
            label = class_names_label[folder]

            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)

                # Open and resize the img
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # Append the image and its corresponding label to the output
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')

        output.append((images, labels))

    return output
#Mostra un'immagine casuale con label del presente nel dataset
def display_random_image(class_names, images, labels):
    """
        Display a random image from the images array and its correspond label from the labels array.
    """

    index = np.random.randint(images.shape[0])
    plt.figure()
    plt.imshow(images[index])
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image #{} : '.format(index) + class_names[labels[index]])
    plt.show()
#Mostra esempi di immagini fornite per il training
def display_examples(class_names, images, labels):
    """
        Display 25 images from the images array with its corresponding labels
    """

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Some examples of images of the dataset", fontsize=16)
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()

def plot_accuracy_loss(history):
    """
        Plot the accuracy and the loss during the training of the nn.
    """
    fig = plt.figure(figsize=(10,5))

    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


#######Model configuration###########
#Dimensione scelta per le immagini
IMAGE_SIZE = (150, 150)
#Numero di campioni propagati attraverso la rete
batch_size=16
#Cicli di addestramento del modello
epochs=24
#Suddivisione di convalida che aiuta a migliorare le prestazioni del modello dopo ogni epoca
validation_split = 0.2


#Caricamento dei dati di training e test nelle due variabili
(train_images, train_labels), (test_images, test_labels) = load_data(pathTrain, pathTest)
#Randomizzazione delle posizioni dei dati
train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
#Numero dei campiono di training
n_train = train_labels.shape[0]
#Numero dei campioni di test
n_test = test_labels.shape[0]

print ("Number of training examples: {}".format(n_train))
print ("Number of testing examples: {}".format(n_test))
print ("Each image is of size: {}".format(IMAGE_SIZE))

#Conta i campioni presenti per ogni classe
_, train_counts = np.unique(train_labels, return_counts=True)
_, test_counts = np.unique(test_labels, return_counts=True)

#Mostra il grafico con i dati, il numero di campioni per ogni classe
pd.DataFrame({'train': train_counts,
                    'test': test_counts},
             index=class_names
            ).plot.bar()
plt.show()

#Scale e normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0
#display_random_image(class_names, train_images, train_labels)
display_examples(class_names, train_images, train_labels)

#Creazione del modello
model = tf.keras.Sequential([
    #Layer convoluzionale 2D
    tf.keras.layers.Conv2D(
        #Numero di filtri per lo strato convoluzionale (dimensione dell'output)
        filters=64,
        #Specifica l'altezza e la lunghezza della finestra dello strato 2D
        kernel_size= (3, 3),
        #Padding disattivato, inoltre strodes non presenti
        padding= "valid",
        # Funzione di attivazione
        activation = 'relu',
        #Dimensione dell'input
        input_shape = (150, 150, 3)
    ),
    tf.keras.layers.Dropout(0.2),

    #Layer di max pooling
    tf.keras.layers.MaxPooling2D(
        #Dimensione Finestra dove prendere il massimo
        pool_size= (2,2),
        #Finestra di spostamento per ogni pooling step
        strides= (2,2),
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="valid",
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        padding="valid",
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    ),
    # Layer di appiattimento
    tf.keras.layers.Flatten(),
    # Layer di attivazion
    tf.keras.layers.Dense(
        #Dimensione dello spazio d'output
        units=128,
        #Funzione di attivazione
        activation='relu'
    ),
    tf.keras.layers.Dense(
        units=4,
        activation='softmax'
    )
])
#Mostra l'architettura del mdoello
model.summary()

#tf.keras.utils.plot_model(model, to_file="./scheme_model.png", show_shapes=True, show_layer_names=True)

#########################################
#Ottimizzatore che implementa l'algoritmo Adam
#L'ottimizzazione di Adam è un metodo di discesa del gradiente stocastico basato sulla stima adattiva dei momenti del primo e del secondo ordine.
opt = tf.keras.optimizers.Adam(
    learning_rate=0.001, # Tensore di apprendimento che rappresenta il tasso di apprendimento
    beta_1 = 0.9, beta_2 = 0.999, # Tassi di decadimento esponenziale per le stime del 1° e 2° momento
    name="Adam"
)
#Funzione di perdita di crossentropia, da usare quando sono presenti due o più classi di etichette,
#le etichette sono previste essere numeri interi e i valori sono espressi in virgola mobile.
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
)
#Compilazione del modello
model.compile(optimizer = opt, loss = loss,
              metrics=['accuracy', #precisione del mdoello
                       'mae' #Mean Absolut error, metrica di regressione, indica la media della somma tra il valore effettivo e quello previsto.
                    ])
#Addestra il modello, history è un record dei valori di perdita di addestramento e dei valori delle metriche
history = model.fit(train_images, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split = validation_split,
                    shuffle= True #Mescola i dati dell'addestramento prima di ogni epoca
                    )
#Mostra dati andamento addestramento
plot_accuracy_loss(history)

#Procedi con la valutazione del modello
test_loss = model.evaluate(test_images, test_labels)

#Fai una predizione
predictions = model.predict(test_images)
#Prendi la label predetta
pred_labels = np.argmax(predictions, axis = 1)
#Mostra immagine con la label predetta
display_random_image(class_names, test_images, pred_labels)


CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True,
           annot_kws={"size": 10},
           xticklabels=class_names,
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.show()


model.save("saved_model/my_model.h5")

#https://stephenallwright.com/cross_val_score-sklearn/
#cross_val_score in Python
#scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')

#print("Mean score of %0.2f with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

