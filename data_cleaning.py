def trainingsdata_cleaning(path):
    data=[]
    value=[]
    #Discard first Line Intro-Line
    f.readline()
    while(True):
        InfoLine = f.readline()
        if(InfoLine == ''): 
            break
        temp=[]
        for i in InfoLine:
            if(i!='\t'):
                temp.append(ord(i))
            else:
                break;
        value.append(InfoLine[len(InfoLine)-2])
        data.append(temp)

    return value, data

def inputdata_cleaning(path):
        #Here the Input data is Read in and converted to numbers for prediction
    inputdata=[]
    while(True):
        infoLine = inputFile.readline()
        if(infoLine == ''): #if file end is reached end reading-in process
            break
        temp=[]
        for i in infoLine:
            if(i!='\n'):
                temp.append(ord(i))
            else:
                break;
        if(temp!=[]):
            inputdata.append(temp)
    return inputdata

    def normalise_data(data):
    # Normalise features
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data