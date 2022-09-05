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
    items.remove("d")


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
    #classSc = fiveClassScore(dataFrameScore["V15"])
    classSc = foureClassScore(dataFrameScore["V15"])

    print(fr"Le cinque classi selezionate da describe:  {classSc}")
    print(tokenConsole)

    # Costruito il nuovo dataframe con featur "sex","age","score","imageData"
    finalDataFrame = dataFrameBuilderFinal(listOfDataset)
    print(fr"Numero di dati ottenuti: {len(finalDataFrame)}")

    #Da commentare se si è gia diviso il dataset
    fiveClass = DivisionClasses(classSc)
    crea_dataset_partizionato(directoryPath, finalDataFrame,
                              fiveClass
                              , sizeTrain=80)
    print("Inserimeto finito!")


class DivisionClasses:
    def __init__(self, listScoreDivision):
        self.listaScoreDivision = listScoreDivision
    def getClassByScore(self, score):
        current= 1
        for levelScore in self.listaScoreDivision:
            if(score > levelScore):
                current+=1
            else: return str(current)
        return str(len(self.listaScoreDivision))

    def getNumberOfClass(self):
        return len(self.listaScoreDivision)


def crea_dataset_partizionato(directoryPath, finalDataFrame, divisionClass: DivisionClasses, sizeTrain: float =80):
    #####Creazione della struttura del DatasetFinale#####
    directoryPath += "/d"
    os.mkdir(directoryPath)

    dirTest = directoryPath + "/seg_test"
    dirTrain = directoryPath + "/seg_train"

    os.mkdir(dirTest)
    os.mkdir(dirTrain)

    subFolders = [str(x) for x in list(range(1, divisionClass.getNumberOfClass()+1))  ]

    for i in subFolders: os.mkdir(dirTest + "/" + i)
    for i in subFolders: os.mkdir(dirTrain + "/" + i)
    #######################################################
    #####Inserimento delle immagini
    import shutil
    import random

    #Percentuale di composizione del train
    percentage_train = sizeTrain/100
    #Percentuale di composizione del test
    percentage_test = ((100 - sizeTrain)/100)
    #Genera nomi casuali
    rName = randomword
    listPath, listScore = finalDataFrame['Directory ID'], finalDataFrame['ScoreVisit']

    for (path, score) in zip(listPath, listScore):
        path_destination = divisionClass.getClassByScore(score)

        # Prendi tutte le immagini
        files = os.listdir(path)
        for fname in files:
            # Metti nella cartella del train
            val = random.random()
            if val < percentage_train:
                shutil.copy2(os.path.join(path, fname), dirTrain + "/" + path_destination + "/" + rName() + ".png")
            # Metti nella cartella del test
            else:
                shutil.copy2(os.path.join(path, fname), dirTest + "/" + path_destination + "/" + rName() + ".png")

    #######################################################

import random, string
def randomword(length=8):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))


if __name__ == '__main__':
    directoryPath = r"C:\Users\elpid\Desktop\DataSetNew\converted"
    loadDataset(directoryPath)



