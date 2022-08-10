import os
import pydicom
import numpy as np
from PIL import Image
from funzioni_utili.datasetLoader import extractDataForPaziente


def convert_dcm_jpg(name):
    """
    Converte il file .DCM in un file .JPG
    :param name: path del file
    :return: file .JPG convertito
    """
    im = pydicom.dcmread(fp=name)
    im = im.pixel_array.astype(float)
    rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixels
    final_image = np.uint8(rescaled_image)  # integers pixels
    final_image = Image.fromarray(final_image)
    return final_image


def get_names(directoryPath: str):
    """
    Prende le path dei file .DCM
    :param directoryPath: nome della directory che presenta all'intermo i file .DCM
    :return: lista dei file .DCM
    """
    names = []
    nameofFiles = os.listdir(directoryPath)
    for n in nameofFiles:
        names.append(directoryPath + n)

    return names


if __name__ == '__main__':
    directoryPath = "C:/Users/elpid/Desktop/DataSetNew"

    items = os.listdir(directoryPath)
    items.remove("Score.xlsx")
    newpath = directoryPath + "/converted"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i in items:
        if not os.path.exists(newpath + "/" + i):
            os.makedirs(newpath + "/" + i)

    # Prendi i dati dei pazienti
    listOfDataset = []
    for i in items:
        df = extractDataForPaziente(directoryPath, i)
        dirRm = df['Directory ID']
        for dir in dirRm:
            subPath = dir.split("/")
            idPaziente = subPath[-2]
            idDir = subPath[-1]

            path = newpath + "/" + idPaziente + "/" + idDir + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            names = get_names(dir + "/")
            part = 0
            for n in names:
                img = convert_dcm_jpg(n)
                img.save(path + 'part_' + str(part) + '.png')
                part += 1
