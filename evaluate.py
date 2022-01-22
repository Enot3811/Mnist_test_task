import argparse
import pandas as pd
import os

import numpy as np
from sklearn import metrics


def parse_args():

    """
    A function for parsing arguments from command line
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--ground_truth", help="path to csv file with ground truth labels", type=str)
    parser.add_argument("--predictions", help="path to csv file with model predictions", type=str)

    
    args = parser.parse_args()

    correctInput = True

    if args.ground_truth is None or not(os.path.isfile(args.ground_truth)):
        print('Could not find the csv file with ground_truth labels')
        correctInput = False

    if args.predictions is None or not(os.path.isfile(args.predictions)):
        print('Could not find the csv file with model predictions')
        correctInput = False

    return correctInput, args


def evaluate_model(pathToTruth, pathToPredictions):

    """
    Gets two csv files containing ground truth labels and model predictions and evaluate model
    """
    
    truthDf = pd.read_csv(pathToTruth, names=('path', 'label'))
    predDf = pd.read_csv(pathToPredictions, names=('path', 'label'))

    truthArr = np.empty(len(truthDf))
    predArr = np.empty(len(predDf))

    if len(truthDf) != len(predDf):
        print('somethign wrong')

    for i, (truth, pred) in enumerate(zip(truthDf['label'], predDf['label'])):

        truthArr[i] = truth 
        predArr[i] = pred

    accuracy = metrics.accuracy_score(truthArr, predArr)
    confus_matr = metrics.confusion_matrix(truthArr, predArr, labels=[0,1,2,3,4,5,6,7,8,9])

    return accuracy, confus_matr

    



if __name__ == '__main__':

    correctInput, args = parse_args()

    if not correctInput:
        quit()

    accuracy, confus_matr = evaluate_model(args.ground_truth, args.predictions)

    print('Accuracy on the test set is %.3f \n' % (accuracy))
    print('Confusion matrix is\n', confus_matr)

