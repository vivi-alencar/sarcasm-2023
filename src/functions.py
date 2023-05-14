import numpy as np

import math
import gc

import keras
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import *

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# needed to plot training history (loss, acc). History is set as globalon def multiTask_multimodal
import matplotlib.pyplot as plt 

import tensorflow as tf
from . import config

global seed
seed = 1337
np.random.seed(seed)

print('=========================================================')
print('Software versions used:')
print('Tensorflow: ' + tf.__version__)
print('Keras: ' + keras.__version__)

def sarcasm_classification_performance(prediction, test_label):
    """TODO: Add a description
    :param prediction: Describe parameter prediction
    :param test_label: Describe parameter test_label
    :return: Describe the return value
    """
    true_label=[]
    predicted_label=[]

    for i in range(test_label.shape[0]):
        true_label.append(np.argmax(test_label[i]))
        predicted_label.append(np.argmax(prediction[i]))

    accuracy      = accuracy_score(true_label, predicted_label)
    prfs_weighted = precision_recall_fscore_support(true_label, predicted_label, average='weighted')

    return accuracy, prfs_weighted


def attention(x, y):
    """TODO: Add a description
    :param x: Describe parameter x
    :param y: Describe parameter y
    :return: Describe the return value
    """
    m_dash = dot([x, y], axes=[2,2])
    m = Activation('softmax')(m_dash)
    h_dash = dot([m, y], axes=[2,1])
    return multiply([h_dash, x])


def divisorGenerator(n):
    """TODO: Add a description
    :param n: Describe parameter n
    :return: Describe the return value
    """
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


# flake8: noqa: E221 # multiple spaces before operator
def featuresExtraction_original(foldNum, exMode):
    """TODO: Add a description
    :param foldNum: Describe parameter foldNum
    :param exMode: Describe parameter exMode
    :return: Describe the return value
    """
    global train_uVisual, train_uAudio, train_sarcasm_label
    global test_uVisual, test_uAudio, test_sarcasm_label

    sarcasm = np.load('feature_extraction/dataset' + str(exMode) + '_original/sarcasmDataset_speaker_dependent_' + str(exMode) + '.npz',
                      mmap_mode='r',
                      allow_pickle=True)
    # ======================================================
    train_uVisual     = sarcasm['feautesUV_train'][foldNum]
    train_uVisual     = np.array(train_uVisual)
    train_uVisual     = train_uVisual/np.max(abs(train_uVisual))
    # ======================================================
    train_uAudio      = sarcasm['feautesUA_train'][foldNum]
    train_uAudio      = np.array(train_uAudio)
    train_uAudio      = train_uAudio/np.max(abs(train_uAudio))
    # ======================================================
    test_uAudio       = sarcasm['feautesUA_test'][foldNum]
    test_uAudio       = np.array(test_uAudio)
    test_uAudio       = test_uAudio/np.max(abs(test_uAudio))
    # ======================================================
    test_uVisual      = sarcasm['feautesUV_test'][foldNum]
    test_uVisual      = np.array(test_uVisual)
    test_uVisual      = test_uVisual/np.max(abs(test_uVisual))
    # ======================================================
    train_sarcasm_label = sarcasm['feautesLabel_train'][foldNum]
    test_sarcasm_label  = sarcasm['feautesLabel_test'][foldNum]
    

def featuresExtraction_fastext(foldNum, exMode):    
    """TODO: Add a description
    :param foldNum: Describe parameter foldNum
    :param exMode: Describe parameter exMode
    :return: Describe the return value
    """
    global train_uText, train_sentiment_uText_implicit, train_sentiment_uText_explicit, train_emotion_uText_implicit, train_emotion_uText_explicit
    global test_uText, test_sentiment_uText_implicit, test_sentiment_uText_explicit, test_emotion_uText_implicit, test_emotion_uText_explicit

    path = 'feature_extraction/dataset'+str(exMode)+'_fasttext/sarcasmDataset_speaker_dependent_'+str(exMode)+'_'+str(foldNum)+'.npz'
    data = np.load(path, mmap_mode='r')
    # =================================================================
    train_sentiment_uText_implicit = data['train_sentiment_uText_implicit']
    train_sentiment_uText_explicit = data['train_sentiment_uText_explicit']
    train_emotion_uText_implicit   = data['train_emotion_uText_implicit']
    train_emotion_uText_explicit   = data['train_emotion_uText_explicit']
    # =================================================================
    test_emotion_uText_implicit    = data['test_emotion_uText_implicit']
    test_emotion_uText_explicit    = data['test_emotion_uText_explicit']
    test_sentiment_uText_implicit  = data['test_sentiment_uText_implicit']
    test_sentiment_uText_explicit  = data['test_sentiment_uText_explicit']
    # =================================================================
    train_uText          = data['train_uText']
    train_uText          = np.array(train_uText)
    train_uText          = train_uText/np.max(abs(train_uText))
    # =================================================================
    test_uText           = data['test_uText']
    test_uText           = np.array(test_uText)
    test_uText           = test_uText/np.max(abs(test_uText))

        
def multiTask_multimodal(foldNum: int, config: config.Config):
    """TODO: Add a description
    :param config: Describe parameter config
    :return: Describe the return value
    """
    # ===========================================================================================================================================
    in_uText        = Input(shape=(train_uText.shape[1], train_uText.shape[2]), name='in_uText')
    rnn_uText_T     = Bidirectional(GRU(config.r_units(), return_sequences=True, dropout=config.rdrop(), recurrent_dropout=config.rdrop()), merge_mode='concat', name='rnn_uText_T')(in_uText)
    td_uText_T      = Dropout(config.drop())(TimeDistributed(Dense(config.td_units(), activation='relu'))(rnn_uText_T))
    attn_uText      = attention(td_uText_T, td_uText_T)
    rnn_uText_F     = Bidirectional(GRU(config.r_units(), return_sequences=False, dropout=config.rdrop(), recurrent_dropout=config.rdrop()), merge_mode='concat', name='rnn_uText_F')(attn_uText)
    td_uText        = Dropout(config.drop())(Dense(config.td_units(), activation='relu')(rnn_uText_F))
    # ===========================================================================================================================================
    in_uVisual      = Input(shape=(train_uVisual.shape[1],), name='in_uVisual')
    td_uVisual      = Dropout(config.drop())(Dense(config.td_units(), activation='relu')(in_uVisual))
    # ===========================================================================================================================================
    in_uAudio       = Input(shape=(train_uAudio.shape[1],), name='in_uAudio')
    td_uAudio       = Dropout(config.drop())(Dense(config.td_units(), activation='relu')(in_uAudio))
    print('td_uText: ',td_uText.shape)

    # ===========================================================================================================================================
    # =================================== internal attention (multimodal attention) =============================================================
    # ===========================================================================================================================================
    if td_uVisual.shape[1]%config.numSplit() == 0:
        td_text   = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/config.numSplit()),config.numSplit())))(td_uText)
        td_visual = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/config.numSplit()),config.numSplit())))(td_uVisual)
        td_audio  = Lambda(lambda x: K.reshape(x, (-1, int(int(x.shape[1])/config.numSplit()),config.numSplit())))(td_uAudio)
        print('td_text: ',td_text.shape)
        print('td_visual: ',td_visual.shape)
        print('td_audio: ',td_audio.shape)

        intAttn_tv = attention(td_text, td_visual)
        intAttn_ta = attention(td_text, td_audio)
        intAttn_vt = attention(td_visual, td_text)
        intAttn_va = attention(td_visual, td_audio)
        intAttn_av = attention(td_audio, td_visual)
        intAttn_at = attention(td_audio, td_text)

        intAttn = concatenate([intAttn_tv, intAttn_ta, intAttn_vt, intAttn_va, intAttn_av, intAttn_at], axis=-1)
        print('intAttn: ', intAttn.shape)

    else:
        print('choose numSplit from '+ str(list(map(int, divisorGenerator(int(td_uVisual.shape[1])))))+'')
        return

    # ===========================================================================================================================================
    # =================================== external attention (self attention) ===================================================================
    # ===========================================================================================================================================
    extCat  = concatenate([td_text, td_visual, td_audio], axis=-1)
    extAttn = attention(extCat, extCat)
    print(extAttn.shape)
    # ===========================================================================================================================================
    merge_inAttn_extAttn  = concatenate([td_text, td_visual, td_audio, intAttn, extAttn], axis=-1)
    merge_inAttn_extAttn = Dropout(config.drop())(Dense(config.td_units(), activation='relu')(merge_inAttn_extAttn))
    print(merge_inAttn_extAttn.shape)
    # ===========================================================================================================================================
    merge_rnn  = Bidirectional(GRU(config.r_units(), return_sequences=False, dropout=config.rdrop(), recurrent_dropout=config.rdrop()), merge_mode='concat', name='merged_rnn')(merge_inAttn_extAttn)
    merge_rnn = Dropout(config.drop())(Dense(config.td_units(), activation='relu')(merge_rnn))
    print(merge_rnn.shape)
    # ===========================================================================================================================================
    output_sarcasm = Dense(2, activation='softmax', name='output_sarcasm')(merge_rnn) # print('output_sarcasm: ',output_sarcasm.shape)
    # ===========================================================================================================================================
    output_senti_implicit = Dense(3, activation='softmax', name='output_senti_implicit')(merge_rnn) # print('output_senti_implicit: ',output_senti_implicit.shape)
    # ===========================================================================================================================================
    output_senti_explicit = Dense(3, activation='softmax', name='output_senti_explicit')(merge_rnn) # print('output_senti_explicit: ',output_senti_explicit.shape)
    # ===========================================================================================================================================
    output_emo_implicit = Dense(9, activation='sigmoid', name='output_emo_implicit')(merge_rnn) # print('output_emo_implicit: ',output_emo_implicit.shape)
    # ===========================================================================================================================================
    output_emo_explicit = Dense(9, activation='sigmoid', name='output_emo_explicit')(merge_rnn) # print('output_emo_explicit: ',output_emo_explicit.shape)
    # ===========================================================================================================================================
    model = Model(inputs=[in_uText, in_uAudio, in_uVisual],
                    outputs=[output_sarcasm, output_senti_implicit, output_senti_explicit, output_emo_implicit, output_emo_explicit])
    model.compile(loss={'output_sarcasm':'categorical_crossentropy',
                        'output_senti_implicit':'categorical_crossentropy',
                        'output_senti_explicit':'categorical_crossentropy',
                        'output_emo_implicit':'binary_crossentropy',
                        'output_emo_explicit':'binary_crossentropy'},
                    sample_weight_mode='None',
                    optimizer='adam',
                    metrics={'output_sarcasm':'accuracy',
                            'output_senti_implicit':'accuracy',
                            'output_senti_explicit':'accuracy',
                            'output_emo_implicit':'accuracy',
                            'output_emo_explicit':'accuracy'})
    print(model.summary())

    ###################### model training #######################
    # TODO Initialize random to allow for consistent results for now
    np.random.seed(1)

    path = ('weights/' + config.filePath(long=True) + '_' +
            'numFold_' + str(foldNum) +
            '.hdf5'
    )

    earlyStop_sarcasm = EarlyStopping(monitor='val_output_sarcasm_loss', patience=30)
    bestModel_sarcasm = ModelCheckpoint(path, monitor='val_output_sarcasm_acc', verbose=1, save_best_only=True, mode='max')

    history = model.fit([train_uText, train_uAudio, train_uVisual], 
                        [train_sarcasm_label, train_sentiment_uText_implicit, train_sentiment_uText_explicit, train_emotion_uText_implicit, train_emotion_uText_explicit],
                        epochs=config.numEpochs(),
                        batch_size=32,
                        # sample_weight=train_mask_CT,
                        shuffle=True,
                        callbacks=[earlyStop_sarcasm, bestModel_sarcasm],
                        validation_data=([test_uText, test_uAudio, test_uVisual], [test_sarcasm_label,test_sentiment_uText_implicit, test_sentiment_uText_explicit,test_emotion_uText_implicit, test_emotion_uText_explicit]),
                        verbose=1)
    
    model.save_weights(path)
    #model.load_weights(path)
    prediction = model.predict([test_uText, test_uAudio, test_uVisual])
    
    performance = sarcasm_classification_performance(prediction[0], test_sarcasm_label)
    print(performance)
    
    # summarize history for accuracy
    plt.plot(history.history['output_sarcasm_accuracy'])
    plt.plot(history.history['val_output_sarcasm_accuracy'])
    plt.title('model output_sarcasm_accuracy')
    plt.ylabel('output_sarcasm_accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['output_sarcasm_loss'])
    plt.plot(history.history['val_output_sarcasm_loss'])
    plt.title('model output_sarcasm_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Write log file
    log = (path + '\n' + 
           '=============== sarcasm ===============\n' +
           'loadAcc: '+ str(performance[0]) + '\n' +
           'prfs_weighted: '+ str(performance[1]) + '\n'*2
    )
    open('results/' + config.filePath() + '.txt', 'a').write(log)
    
    ################### release gpu memory ###################
    K.clear_session()
    del model      
    gc.collect()
