import csv
import os
import sys
import pandas as pd


def make_csv(pathToDir, nameCsv):

    dirs_classes = os.listdir(pathToDir)

    with open(nameCsv, 'w', newline='') as csvfile:

        csvWriter = csv.writer(csvfile, delimiter=',')
   
        for dir_class in dirs_classes:

            pathToImages = os.path.join(pathToDir, dir_class)
            images = os.listdir(pathToImages)

            for image in images:
                
                csvWriter.writerow([
                    os.path.join(pathToImages, image),
                    dir_class
                    ])


def read_csv(pathToCsv):

    df = pd.read_csv(pathToCsv, names=['path', 'label'])

    return df


if __name__ == '__main__':

    make_csv(r'mnist_png\training', 'train.csv')
    make_csv(r'mnist_png\testing', 'test.csv')
    #df = read_csv('test.csv')

    #print(df['path'][0:2])
    #for i in range(len(df)):
    #    path = df['path'][i]
    #    path = path.replace('\\', '/')
        
    #    df.at[i,'path'] = path


    
    
    #print(df['path'][0:2])

    
