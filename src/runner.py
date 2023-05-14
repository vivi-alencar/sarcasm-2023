if __name__ == "__main__":
    import sys
    print('Error: Please run this file from one of the Jupyter notebooks provided.')
    sys.exit()

from . import functions
from . import config

import time


def start(config: config.Config):
    """runner entry point
    :param config: Configuration object
    """
    log = '#' * 150 + '\n' + time.ctime() + '\n'  # 'Mon Oct 18 13:35:29 2010''
    open('results/' + config.filePath() + '.txt', 'a').write(log)
    for foldNum in config.numFolds():
        # it has all the modalities features based on averaging
        functions.featuresExtraction_original(foldNum, config.speaker_dependent())

        # it has only text word-wise features using fasttext with sentiment and emotion label
        functions.featuresExtraction_fastext(foldNum, config.speaker_dependent())

        functions.multiTask_multimodal(foldNum, config)
