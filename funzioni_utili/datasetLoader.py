import os
import pandas as pd
from funzioni_utili.statisticFunctions import *

def extractDataForPaziente(directoryPath, idPaziente):
    """
    Prendi il csv di un paziente e inserisci in prima colonna
    il path della cartella dove trovare le risonanze per quella riga
    :param idPaziente: nome della cartella dove sono contenute le info del paziente
    :return: dataframe del paziente
    """
    filePazienteToOpen = directoryPath + "/" + idPaziente + "/" + idPaziente + ".xlsx"
    df = pd.DataFrame(pd.read_excel(filePazienteToOpen))
    listOfDir = df['Image Data ID']
    # print(df['Image Data ID'][1])
    count = 0
    list = []
    for f in listOfDir:
        list.append(directoryPath + "/" + idPaziente + "/" + f)

    list2 = []
    for i in range (len(list)):
        list2 = idPaziente

    df.insert(0, "Directory ID", list)
    df.insert(0, "Paziente ID", list2)

    #Drop di colonne inutili
    df = df.drop("Modality", axis=1)
    df = df.drop("Type", axis=1)
    df = df.drop("Group", axis=1)
    df = df.drop("Image Data ID", axis=1)
    return df
def takeScoreDataAsDataFrame(directoryPath):
    """
    Prende il csv degli score dei pazienti
    :param directoryPath: path del dataset
    :return: dataframe degli Score
    """
    pathScore = "/Score.xlsx"
    df = pd.DataFrame(pd.read_excel(directoryPath + pathScore))
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(df)
    return df
def dataFrameBuilderFinal(listOfData):
    finalListData = []
    for i in listOfData:
        num = len(i)
        for y in range(num):
            finalListData.append(i.loc()[y])

    final_df = pd.DataFrame(finalListData)
    final_df.drop(["Visit", "Acq Date", "Format", "Description"], axis=1, inplace=True)
    print(fr"Features del dataset: {final_df.columns}")
    print(fr"Dataframe pronto per il CNN")

    return final_df

def loadDataset(directoryPath):
    tokenConsole = "|||"*50
    print(tokenConsole)
    print(fr"Directory indicata per il dataset --> {directoryPath}")

    #DataScore dei pazienti "File Score.xlxs"
    dataFrameScore = takeScoreDataAsDataFrame(directoryPath)

    #Lista delle directory dei pazienti
    items = os.listdir(directoryPath)
    items.remove("Score.xlsx")


    #Prendi i dati dei pazienti
    listOfDataset = []
    for i in items:
        df = extractDataForPaziente(directoryPath, i)
        listOfDataset.append(df)

    for paziente in listOfDataset:
        listDelPaziente=[]
        listDelPaziente = paziente["Paziente ID"]

    for paziente in listOfDataset:
        scoreVisite = []
        listanumeroVisite = paziente['Visit']
        numeroPaziente = paziente['Paziente ID'][0]
        rowDataScores = dataFrameScore.loc[dataFrameScore['Unnamed: 0'] == int(numeroPaziente)]

        for i in listanumeroVisite:
            if (i<10): numbVisit = "V{}{}".format("0",i)
            else: numbVisit = "V{}".format(i)
            scoreVisite.append(int(rowDataScores[numbVisit]))
        paziente["ScoreVisit"]=scoreVisite


    #Statistiche del dataset
    countMale, countFemale = countMaleAndFemaleInDataset(listOfDataset)
    print(fr"Numero di uomini: {countMale}, numero di donne: {countFemale}")
    min,max,average = min_max_average_age_InDataset(listOfAgeInDataset(listOfDataset))
    print(fr"Età minima: {min}, Età massima: {max}, Età media: {average}")
    #Prendiamo ultima colonna degli score finai
    #Si è selezionato min 33% 50% 66% max
    #Per le classi di gravità del Parkinson
    fiveClass = fiveClassScore(dataFrameScore["V15"])
    print(fr"Le cinque classi selezionate da describe:  {fiveClass}")
    print(tokenConsole)

    # Costruito il nuovo dataframe con featur "sex","age","score","imageData"
    finalDataFrame = dataFrameBuilderFinal(listOfDataset)
    print(fr"Numero di dati ottenuti: {len(finalDataFrame)}")

    #Da commentare se si è gia diviso il dataset
    #crea_tre_parti_dataset(directoryPath, finalDataFrame, fiveClass)
    print("Inserimeto finito!")



def crea_tre_parti_dataset(directoryPath, finalDataFrame, fiveClass):
    #####Creazione della struttura del DatasetFinale#####
    directoryPath += "/d"
    os.mkdir(directoryPath)

    dirPred = directoryPath + "/seg_pred"
    dirTest = directoryPath + "/seg_test"
    dirTrain = directoryPath + "/seg_train"

    os.mkdir(dirPred)
    os.mkdir(dirTest)
    os.mkdir(dirTrain)

    subFolders = ["1", "2", "3", "4", "5"]
    for i in subFolders: os.mkdir(dirTest + "/" + i)
    for i in subFolders: os.mkdir(dirTrain + "/" + i)
    #######################################################
    #####Inserimento delle immagini
    import shutil
    import random
    percentage_test = 0.25  # percentuale di composizione del test
    percentage_train = 0.65  # percentuale di composizione del train
    percentage_pred = 0.10  # percentuale per la validazione
    rName = randomword
    listPath, listScore = finalDataFrame['Directory ID'], finalDataFrame['ScoreVisit']

    for (path, score) in zip(listPath, listScore):
        # A seconda dello score cambia la destinazione
        path_destination = ""
        if (score <= fiveClass[0]):
            path_destination = "1"
        elif (score <= fiveClass[1]):
            path_destination = "2"
        elif (score <= fiveClass[2]):
            path_destination = "3"
        elif (score <= fiveClass[3]):
            path_destination = "4"
        else:
            path_destination = "5"
        # Prendi tutte le immagini
        files = os.listdir(path)
        for fname in files:
            # Metti nella cartella del train
            val = random.random()
            if val > percentage_train:
                shutil.copy2(os.path.join(path, fname), dirTrain + "/" + path_destination + "/" + rName() + ".png")
            elif val > percentage_test:
                shutil.copy2(os.path.join(path, fname), dirTest + "/" + path_destination + "/" + rName() + ".png")
            else:
                shutil.copy2(os.path.join(path, fname), dirPred + "/" + rName() + ".png")
    #######################################################

import random, string
def randomword(length=8):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


if __name__ == '__main__':
    directoryPath = r"C:\Users\elpid\Desktop\DataSetNew\converted"
    loadDataset(directoryPath)



