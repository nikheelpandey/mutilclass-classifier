import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def createCSVFromFolder(path, eval=False):

    file_names = []
    file_class = []
    classes = os.listdir(path)
    
    for cl in classes:
        files = os.listdir(os.path.join(path, cl))
        files = [os.path.abspath(os.path.join(path, cl, f)) for f in files]
        filecls = [cl for i in range(len(files))]
        file_names.extend(files)
        file_class.extend(filecls)

    data = {
        'file_path':file_names,
        'category':file_class
    }
    

    df = pd.DataFrame(list(zip(file_names, file_class)), columns = ['file_path', 'category'])

    df.to_csv('dataset.csv', index=False)

    cats = {}
    
    for i  in range(len(df['category'].unique())):
        cats[df['category'].unique()[i]] = int(i)

    if not eval:

        with open('categories.json','w') as f:
            json.dump(cats, f)


    return df
    

def getTrainTestSplit(df, split=0.2):
    return train_test_split( df, test_size=split, random_state=42, shuffle=True)

    



if __name__=='__main__':

    path = '../dogs-vs-cats/'
    df = createCSVFromFolder(path)
    train, test = getTrainTestSplit(df)
    print(train)