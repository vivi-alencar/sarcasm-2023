import itertools

class Config:
    #region Static configuration that is used for every run
    _drop = 0.3
    _rdrop = 0.3
    _r_units = 300
    _td_units = 50

    # numSplits depends on speaker dependency
    _numSplit = {
        True: 50,
        False: 25
    }

    # foldNums depends on speaker dependency
    _numFolds = {
        True: [0, 1, 2, 3, 4],
        False: [3]
    }
    
    #endregion

    #def __init__(self, speaker_dependent: bool, modalities: str = 'tav', num_epochs: int = 200):
    def __init__(self, speaker_dependent: bool, num_epochs: int = 200):
        '''Constructor
        :param speaker_dependent: True or False
        :param modalities: Modalities to use. Default 'tav' for text, audio and video
        "param num_epochs: How many epochs. Default 200
        '''
        if type(speaker_dependent) is not bool:
            raise ValueError('speaker_dependent must be a boolean value')

        self._speaker_dependent = speaker_dependent
        
        # Only do TAV for now, don't have it configurable
        modalities = 'tav' 
        if not set(modalities) <= set('tavTAV'):
            raise ValueError('Invalid value for modality used, allowed is any combination of tav')
        self._modalities = []
        if 't' in modalities: self._modalities.append('text')
        if 'a' in modalities: self._modalities.append('audio')
        if 'v' in modalities: self._modalities.append('video')

        if type(num_epochs) is not int or num_epochs <= 0:
            raise ValueError('Number of epochs must be a positive integer')
        self._num_epochs = num_epochs

    def speaker_dependent(self): return self._speaker_dependent
    def drop(self): return self._drop
    def rdrop(self): return self._rdrop
    def r_units(self): return self._r_units
    def td_units(self): return self._td_units

    def numSplit(self) -> int:
        return self._numSplit[self._speaker_dependent]

    def numFolds(self) -> int:
        return self._numFolds[self._speaker_dependent]
    
    def numEpochs(self): return self._num_epochs

    # flake8: noqa: W504 # line break after binary operator
    def filePath(self, long = False) -> str:
        modality = '_'.join(self._modalities)
        filePath = ('sarcasm_speaker_dependent_wse_' + str(self.speaker_dependent()) + '_' +
                    modality
        )
        if long:
            filePath += ('_' +
                         'epochs_' + str(self.numEpochs()) + '_' + 
                         'drop_' + str(self.drop()) + '_' +
                         'rdrop_' + str(self.rdrop()) + '_' +
                         'runits_' + str(self.r_units()) + '_' +
                         'tdunits_' + str(self.td_units()) + '_' +
                         'numSplit_' + str(self.numSplit())
            )
        return filePath
    
    def __str__(self):
        value = repr(self) + "\n"
        value += "Drop: " + str(self.drop()) + "\n"
        value += "rDrop: " + str(self.rdrop()) + "\n"
        value += "rUnit: " + str(self.r_units()) + "\n"
        value += "tdUnit: " + str(self.td_units()) + "\n"
        value += "numSplit: " + str(self.numSplit()) + "\n"
        value += "numFolds: " + str(self.numFolds()) + "\n"
        value += "speakerDependent: " + str(self.speaker_dependent()) + "\n"
        value += "modalities: " + str(self._modalities) + "\n"
        value += "num_epochs: " + str(self.numEpochs()) + "\n"        
        return value
                                
