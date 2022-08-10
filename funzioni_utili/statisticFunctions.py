


def fiveClassScore(columnScore):
    frequenzeScore = columnScore.describe(percentiles=[.33, .66, .100])
    # Si è selezionato min 33% 50% 66% max
    # Per le classi di gravità del Parkinson
    five_class_score = [frequenzeScore['min'],
                        frequenzeScore['33%'], frequenzeScore['50%'], frequenzeScore['66%'], frequenzeScore['max']]
    return five_class_score

def countMaleAndFemaleInDataset(dataFrame):
    """
    Conta il numero di uomini e donne nel dataset
    :param dataFrame:
    :return:
    """
    male = female = 0
    for item in dataFrame:
        if item.Sex[0] == "M":
            male+=1
        else:
            female+=1
    return [male,female]
def listOfAgeInDataset(dataFrame):
    """
    Lista le età nel dataset
    :param dataFrame:
    :return:
    """
    list = []
    for item in dataFrame:
        list.append(item.Age[0])
    return list
def min_max_average_age_InDataset(list):
    return min(list), max(list), sum(list)/len(list)
