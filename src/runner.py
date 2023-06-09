if __name__ == "__main__":
    import sys
    print('Error: Please run this file from one of the Jupyter notebooks provided.')
    sys.exit()

from . import functions
from . import config

import time
import numpy


# flake8: noqa: W504 # line break after binary operator
def start(config: config.Config):
    """runner entry point
    :param config: Configuration object
    """
    log = '#' * 150 + '\n' + time.ctime() + '\n'  # 'Mon Oct 18 13:35:29 2010''
    open('results/' + config.filePath() + '.txt', 'a').write(log)

    averageAccuracy = []  
    averagePrecision = []
    averageRecall = []
    averageF1score = []

    for foldNum in config.numFolds():
        # it has all the modalities features based on averaging
        functions.featuresExtraction_original(foldNum, config.speaker_dependent())

        # it has only text word-wise features using fasttext with sentiment and emotion label
        functions.featuresExtraction_fastext(foldNum, config.speaker_dependent())

        performance = functions.multiTask_multimodal(foldNum, config)
        averageAccuracy.append(performance['loadAccuracy'])
        averagePrecision.append(performance['precision'])
        averageRecall.append(performance['recall'])
        averageF1score.append(performance['f1score'])

    if len(config.numFolds()) > 1:
        log = ('Average values for all folds:\n'
            f'  Accuracy: {numpy.mean(averageAccuracy)}\n' +
            f'  Precision: {numpy.mean(averagePrecision)}\n' +
            f'  Recall: {numpy.mean(averageRecall)}\n' +
            f'  F1score: {numpy.mean(averageF1score)}\n\n'
        )
        open('results/' + config.filePath() + '.txt', 'a').write(log)
