import numpy as np
import pickle as pck
from sklearn.metrics import *
import librosa as lb
from librosa.feature import melspectrogram
import librosa.util
from librosa.display import specshow
import os.path as osp
from librosa.filters import get_window
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, GRU, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, MaxPooling1D, Flatten, Activation
from collections import OrderedDict
import yaml
import logging
import gci_utils as gu
import gci_scorer as gs
from pm import Pm


# Prepare logger
logger = logging.getLogger('utils')

def f1_score(precision, recall):
    return 2*(precision*recall)/(precision + recall)


def proba2classes(yproba):
    ycls = np.zeros(yproba.shape, dtype='int8')
    ycls[yproba > 0.5] = 1
    return ycls


def get_scoring_func(scr_str):
    if scr_str == 'f1':
        return f1_score
    elif scr_str == 'precision':
        return precision_score
    elif scr_str == 'recall':
        return recall_score
    elif scr_str == 'roc_auc':
        return roc_auc_score
    elif scr_str == 'balanced_accuracy':
        return balanced_accuracy_score
    elif scr_str == 'average_precision':
        return average_precision_score
    else:
        raise ValueError("Scoring function for '"+scr_str+"' is not defined")


def get_best_epoch_idx(history, metric='val_loss'):
    if 'loss' in metric:
        return int(np.argmin(history.history[metric]))
    else:
        return int(np.argmax(history.history[metric]))


def match_metrics_and_scores(metrics_names, scores):
    res = {}
    for m, s in zip(metrics_names, scores):
        res[m] = s
    return res


# Supported window functions
WINFUNC = {None: None,
           'hamming': np.hamming,
           'hanning': np.hanning,
           'blackman': np.blackman,
           'bartlett': np.bartlett
}


# Calculate spectrogram for a wav audio file
def specgram(s, n_fft, n_hop, window='hann', to_db=True):
    S = np.abs(lb.stft(s, n_fft=n_fft, hop_length=n_hop, window=window))
    if to_db:
        S = lb.amplitude_to_db(S, ref=np.max)
    # The spectogram outputs (freqs, timesteps) and we want (timesteps, freqs) to input into the model
    return S.swapaxes(0, 1)


# Calculate mel-spectrogram for a wav audio file
def melspecgram(s, fs, n_fft, n_hop, window='hann', n_mels=128, to_db=True):
    M = melspectrogram(s, fs, n_fft=n_fft, hop_length=n_hop, window=window, n_mels=128)
    if to_db:
        M = lb.power_to_db(M, ref=np.max)
    # The mel spectogram outputs (freqs, timesteps) and we want (timesteps, freqs) to input into the model
    return M.swapaxes(0, 1)


# Window signal
def frame(s, n_fft, n_hop, window='hann'):
    # Pad the time series so that frames are centered
    s = np.pad(s, (int(n_fft//2), 0), mode='reflect')
    return lb.util.frame(s, n_fft, n_hop, axis=0)*get_window(window, n_fft, False)


# Assign targets to frames
def targets2frames(fname, fs, hop_size, n_frames, n_neighbors=0):
    y = np.zeros((n_frames, 1), int)
    # data = fs*np.loadtxt(fname, usecols=[0])
    # frame_ixs = np.rint(data/hop_size).astype('int')
    data = np.loadtxt(fname, usecols=[0])
    frame_ixs = lb.time_to_frames(data, fs, hop_size)
    # Mark 1 the current frame and potencially also neighboring frames
    for n in range(-n_neighbors, n_neighbors+1):
        y[frame_ixs+n] = 1
    return y


# Pad time steps to a given length
def pad_timesteps(data, max_timesteps, value=0):
    modulo = data.shape[0]%max_timesteps
    pad_width = 0 if modulo == 0 else max_timesteps - modulo
    data = np.pad(data, ((0, pad_width), (0, 0)), constant_values=value)
    return data, pad_width


# Reshape data to (samples, timesteps, features)
def reshape(data, max_timesteps=None, pad_value=0):
    n_timesteps, n_feats = data.shape
    if max_timesteps is None:
         max_timesteps = n_timesteps
    # Pad time steps and return the padded length
    data, pad_width = pad_timesteps(data, max_timesteps, pad_value)
    # Reshape to (samples, timesteps, features)
    data = data.reshape((data.shape[0]//max_timesteps, max_timesteps, n_feats))
    return data, pad_width


# Create CNN layers
def create_cnn_layers(X, n_blocks, n_convs, filters, kernel_size, strides, padding, pool_size, bn=True):
    # CNN blocks
    for ib in range(n_blocks):
        for ic in range(n_convs):
            X = TimeDistributed(Conv1D(filters[ib], kernel_size[ib], strides[ib], padding[ib], activation='relu'))(X)
            if bn:
                # batch normalization
                X = TimeDistributed(BatchNormalization())(X)
            X = Activation('relu')(X)  
    X = TimeDistributed(MaxPooling1D(pool_size=pool_size[ib]))(X)
    return TimeDistributed(Flatten())(X)

# Create RNN layers
def create_rnn_layers(X, rnn_cells, dropout, bn=True, rnn_type='lstm'):    
    # RNN steps
    # X = Masking(mask_value=pad_value)(X)
    for ix in range(len(rnn_cells)):
        if rnn_type == 'lstm':
            X = Bidirectional(LSTM(rnn_cells[ix], return_sequences=True))(X) # LSTM
        elif rnn_type == 'gru':
            X = Bidirectional(GRU(rnn_cells[ix], return_sequences=True))(X) # GRU
        else:
            raise SystemExit('Unknown RNN type: ', rnn_type)
        if bn:
            X = BatchNormalization()(X) # batch normalization
        X = Dropout(dropout[ix])(X)     # dropout

    return X

# Create model
def create_model(input_shape, conv_blocks, n_convs, filters, kernel_size, strides, padding, pool_size,
                 rnn_cells, dropout, bn=True, rnn_type='lstm'):
    # Prepare input
    X_input = Input(shape=input_shape)
    # Create CNN layers
    X = create_cnn_layers(X_input, conv_blocks, n_convs, filters, kernel_size, strides, padding, pool_size, bn)
    # Create RNN layers
    X = create_rnn_layers(X, rnn_cells, dropout, bn, rnn_type)
    # Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X) # time distributed (sigmoid)
    return Model(inputs=X_input, outputs=X)


# Fit Keras model
# noinspection PyPep8Naming
def fit_model(model, X_trn, y_trn, batch_size=None, epochs=1, verbose=1, validation_data=None,
              best_weight_fn=None, early_stop_patience=None, csv_logger=None):
    # Early stopping: None means no stopping => patience is the same as no. of epochs
    early_stop_patience = epochs if early_stop_patience is None else early_stop_patience
    es = EarlyStopping(monitor='val_loss', patience=early_stop_patience)
    callbacks = [es]
    # Init callbacks with saving the best model
    if best_weight_fn:
        callbacks.append(ModelCheckpoint(best_weight_fn, monitor='val_loss', verbose=verbose, save_best_only=True, save_weights_only=True))
    if csv_logger:
        callbacks.append(CSVLogger(csv_logger, separator=';', append=True))
    # Fit the model
    history = model.fit(X_trn, y_trn, epochs=epochs, batch_size=batch_size, validation_data=validation_data,
                        verbose=verbose, callbacks=callbacks)
    # Analyze early stopping:
    # stopped epoch 0 means that no stopping was applied => all epochs were used
    stopped_epoch = es.stopped_epoch if es.stopped_epoch > 0 else epochs
    best_epoch_idx = get_best_epoch_idx(history, 'val_loss')

    return history, stopped_epoch, best_epoch_idx


# Finalize Keras model
# noinspection PyPep8Naming
def finalize_model(model, X, y, batch_size=None, epochs=1, verbose=1, csv_logger=None):
    # Init callbacks with saving the best model
    callbacks = []
    # if tb_logdir is not None:
    #     callbacks.append(TensorBoard(log_dir=tb_logdir, write_graph=True))
    if csv_logger is not None:
        callbacks.append(CSVLogger(csv_logger, separator=';', append=True))
    # Fit the model
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
    return model, history


# Evaluate Keras model using GCI detection measures
def evaluate_model(model, testset, spc_dir, pm_dir, freq_samp, n_timesteps, fft_len, hop_len=0.002,
                           window='boxcar', pad_value=0, sync=None, abs_dist=0.00025, rel_dist=10, min_t0=0.002):
    results = {}
    # Read ground-truth pitch-mark objects
    refr_pm = [Pm(osp.join(pm_dir, fn+'.pm')) for fn in testset]
    # Detect pitch-marks
    pred_pm, _ = detect(model, testset, spc_dir, freq_samp, n_timesteps, fft_len, hop_len,
                        window, pad_value=pad_value, sync=sync)
    # Evaluate
    scorer_rel, scorer_abs = gs.evaluate(zip(refr_pm, pred_pm), abs_dist, rel_dist, min_t0)
    # Extract GCI detection measures
    results['IDR'] = scorer_rel.identification_rate_score()
    results['MR'] = scorer_rel.miss_rate_error()
    results['FAR'] = scorer_rel.false_alarm_rate_error()
    results['IDA'] = scorer_rel.identification_accuracy_error()
    results['A{}'.format(int(100000*abs_dist))] = scorer_abs.identification_accuracy_score()
    results['E{}'.format(rel_dist)] = scorer_rel.accuracy_score()
    return results

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


# Prepare data for input
def prep4input(utt_list, spc_dir, pm_dir=None, fs=16000, n_timesteps=500, fft_len=.016,
               hop_len=.002, window='boxcar', pad_value=0, incl_signal=False):
    data, targets = [], []
    utt_info = {}
    # Convert times to samples
    n_fft = lb.time_to_samples(fft_len, fs)
    n_hop = lb.time_to_samples(hop_len, fs)
    logger.debug('Window length/hop: {}/{} ms ({}/{} samples)'. format(fft_len, hop_len, n_fft, n_hop))
    for u in utt_list:
        # Read speech data
        s_src, fs_src = lb.load(osp.join(spc_dir, u+'.wav'), None)
        s = resample(s_src, fs_src, fs) if fs_src != fs else s_src
        logger.debug('{}: input FS={} Hz, detection FS={} Hz'.format(u, fs_src, fs))
        # Frame the signal
        S = frame(s, n_fft, n_hop, window)
        n_frames = S.shape[0]
        S, pad_width = reshape(S, n_timesteps, pad_value)
        data.append(S)
        logger.debug('{}: {} samples, {} frames, {} padded frames'. format(u, len(s_src), n_frames, pad_width))
        # Add utterance info
        utt_info[u] = {'samp_freq': fs_src, 'n_frames': n_frames, 'pad_width': pad_width}
        if incl_signal:
            utt_info[u]['signal'] = s_src
        if pm_dir:
            # Convert GCI times to frames
            y = targets2frames(osp.join(pm_dir, u+'.pm'), fs, n_hop, n_frames)
            logger.debug('{}: ground-truth GCIs/non-GCIs{}/{}'.format(u, len(y[y==1]), len(y[y==0])))
            y, _ = reshape(y, n_timesteps, pad_value)
            targets.append(y)
    # Stack data
    data = np.vstack(data)
    # Reshape to meet CNN layer requirements
    data = data.reshape(data.shape[0], n_timesteps, n_fft, 1)
    
    return data, np.vstack(targets) if pm_dir else None, utt_info


# Predict GCIs
def predict(model, utt_list, spc_dir, fs=16000, n_timesteps=500, fft_len=.016, hop_len=.002,
            window='boxcar', pad_value=0, incl_signal=False):
    X, _, utt_info = prep4input(utt_list, spc_dir, None, fs, n_timesteps, fft_len, hop_len,
                                window, pad_value, incl_signal=incl_signal)
    logger.debug('Data shape: {}'.format(X.shape))
    Y = model.predict(X)
    logger.debug('Prediction shape: {}'.format(Y.shape))
    return Y, utt_info


# Detect GCIs
def detect(model, utt_list, spc_dir, fs=16000, n_timesteps=500, fft_len=.016, hop_len=.002,
           window='boxcar', pad_value=0, sync=None, incl_signal=False):
    pms = []      # list of predicted pitch-mark objects
    utt_info = {} # utterance info
    # Process syncing information
    if sync is None:
        lsync, rsync = hop_len, hop_len
    elif isinstance(sync, (tuple, list)):
        lsync, rsync = sync[0], sync[1]
    else:
        lsync, rsync = sync, sync
    # Go through all files in the list
    for uname in utt_list:
        logger.debug('Predicting {}'.format(uname))
        # Predict GCIs in shape (samples, timesteps, features)
        Y, utt1_info = predict(model, [uname], spc_dir, fs, n_timesteps, fft_len, hop_len, window,
                               pad_value=pad_value, incl_signal=True)
        # Read speech data to sync to
        s_src, fs_src, n_frames, pad_width = utt1_info[uname]['signal'], utt1_info[uname]['samp_freq'], utt1_info[uname]['n_frames'], utt1_info[uname]['pad_width']
        logger.debug('Signal: FS = {} Hz, len = {} samples ({} frames, {} padded frames)'.format(fs_src, len(s_src), n_frames, pad_width))
        # Flatten predictions and remove padded frames
        y = Y.flatten()[:n_frames]
        # Calculate centers of frames with predicted GCIs
        n_hop = lb.time_to_samples(hop_len, fs_src)
        pred_samples = lb.frames_to_samples(range(n_frames), n_hop) + n_hop//2
        # pred_samples = lb.frames_to_samples(range(n_frames), n_hop)
        # logger.debug('# predicted frames (GCI-frames): {} ({})'.format(len(y), len(y_pred)))
        logger.debug('Prediction on frames: {}'.format(y[:10]))
        logger.debug('Frame samples: {}'.format(pred_samples[:10]))
        # Calculate sync intervals in samples
        n_lsync, n_rsync = lb.time_to_samples(lsync, fs_src), lb.time_to_samples(rsync, fs_src)
        # Convert samples to time
        # pred_times = lb.samples_to_time(
        #     gu.sync_predictions_to_samp_peak(y_pred, pred_samples, s_src, 0, n_hop), fs_src)
        pred_times = lb.samples_to_time(
          gu.sync_predictions_to_samp_peak(y, pred_samples, s_src, n_lsync, n_rsync), fs_src)
        logger.debug('Predicted GCI-frame times: {}'.format(pred_times[:10]))
        # Create PM object and append it to the prediction list
        pms.append(gu.create_gci(pred_times, uname))
        utt_info[uname] = {'samp_freq': fs_src}
        if incl_signal:
            utt_info[uname]['signal'] = s_src
    return pms, utt_info
