# -*- coding: utf-8 -*-
import numpy as np
import logging
from bisect import bisect_left
import warnings
from pm import OnePm, Pm

"""
Glottal closure utilities.

This module contains code for glottal closure instant (GCI) detection. Scikit sklearn library is heavily used.

Attributes:
    logger (:obj:`logging.Logger`): Logger object
    T_MARK_DIST_TIME (float): Tine distance (sec) between "regular" GCI and possibly inserted transitional GCI

"""

logger = logging.getLogger('gci_detect.gci_utils')

# Time distance (s) between "regular" GCI and possibly inserted transitional GCI
T_MARK_DIST_TIME = 0.0005


def create_gci(times, name=None):
    """
    Create GCIs from times in seconds.

    Args:
        times (:obj:`numpy.array`): Array of times in seconds (float) corresponding to GCI placements.
        name (str):                 Name of the sequence

    Returns:
        :obj:`pm.Pm`: Pitch-mark object with GCIs in seconds
    """
    # warnings.warn('create_gci(times) is deprecated, use pm.Pms(times=times) instead', DeprecationWarning)
    gci = Pm()
    gci.set_pmks([OnePm(t, OnePm.type_V) for t in times], name)
    return gci


def insert_trans_gci(gci, min_t0=0.020):
    """Insert transitional GCIs (T-marks) into voiced GCIs (V-marks).

    T-marks delimit voiced regions - a boundary between voiced regions is placed between two V-marks distant at least
    `min_t0`. T-marks are included :attr:`T_MARK_DIST_TIME` before/after a V-mark.

    Args:
        gci (:obj:`pm.Pm`): Pitch-mark object containing individual "voiced" GCIs.
        min_t0 (float):     Minimum T0 value (sec).

    Returns:
        :obj:`pm.Pm`: Pitch-mark object with transitional marks included.

    """
    first, last = gci[0], gci[-1]
    t_pm = OnePm(last.time + T_MARK_DIST_TIME, last.type_T)
    logger.debug('Adding GCI: {}'.format(t_pm))
    gci.append(t_pm)
    pmks = gci.get_all_pmks()
    for idx in range(len(pmks)-2, -1, -1):
        cpm = pmks[idx]
        npm = pmks[idx+1]
        if npm.type != OnePm.type_V:
            continue
        # Detect boundary between two voiced regions
        logger.debug('Current GCI: {}, next GCI: {}, F0={:6.2f}, T0={:8.6f}'.
                     format(cpm, npm, 1/(npm.time-cpm.time) if npm.time != cpm.time else np.inf, npm.time-cpm.time))
        if npm.time - cpm.time > min_t0:
            logger.debug('Voiced region boundary detected: {}-{} (F0={:6.2f}, T0={:8.6f})'.
                         format(cpm.time, npm.time, 1/(npm.time-cpm.time), npm.time-cpm.time))
            # Insert T-marks
            le = OnePm(cpm.time + T_MARK_DIST_TIME, cpm.type_T)
            t = npm.time - T_MARK_DIST_TIME
            ri = OnePm(0 if t < 0 else t, cpm.type_T)
            logger.debug('Adding GCI: {}, {}'.format(le, ri))
            gci.insert(idx+1, [le, ri])
    t = first.time - T_MARK_DIST_TIME
    t_pm = OnePm(0 if t < 0 else t, first.type_T)
    logger.debug('Adding GCI: {}'.format(t_pm))
    gci.insert(0, t_pm)
    return gci


def insert_unvoiced_gci(gci, uttlen, dist=0.004):
    # Set auxiliary transitional mark at the beginning and end of speech signal
    gci.insert(0, OnePm(0, OnePm.type_T))
    gci.append(OnePm(uttlen, OnePm.type_T))
    # Go backwards through all GCIs
    for idx in range(len(gci)-2, -1, -1):
        curr_pm = gci[idx]
        next_pm = gci[idx+1]
        if curr_pm.T and next_pm.T:
            end_time = next_pm.time
            beg_time = 0 if idx == 0 else gci[idx-1].time
            unv_reg_len = end_time - beg_time
            nmarks = int(round(unv_reg_len / dist))
            if nmarks == 0:
                logger.debug('Region {}-{} too short for U-marks => no U-marks inserted'.format(beg_time, end_time))
                continue
            apply_dist = unv_reg_len / nmarks
            logger.debug('Insert {} U-marks: {} - {} with step {}'.format(nmarks, beg_time, end_time, apply_dist))
            unvs = [OnePm(beg_time+t, OnePm.type_U) for t in apply_dist * np.arange(1, nmarks)]
            gci.insert(idx+1, unvs)
    # Remove auxiliary T-marks at the beginning and end of speech signal
    del gci[-1]
    del gci[0]
    return gci


def samples2peak(peaks, samples):
    # Shift samples to the nearest peak
    shifted_saples = set()
    for s in samples:
        pos = bisect_left(peaks, s)
        if pos == 0:
            # Treat very first peak
            p = peaks[0]
        elif pos == len(peaks):
            # Treat very last peak
            p = peaks[-1]
        else:
            # Process inner peaks
            before, after = peaks[pos-1], peaks[pos]
            # Find the closest peak from the two around
            p = after if after-s < s-before else before
        logger.debug('{} shifted to peak sample {}'.format(samp, p))
        samp_shifted.add(p)
    return shifted_samples


def gci2targets(gci, peaks, samp_freq):
    """Shift GCIs to the nearest peak which is taken as target for GCI detection

    Args:
        gci (:obj:`pm.Pm`):         Pitch-mark object with the individual GCIs
        peaks (:obj:`numpy.array`): Array of (negative) peak positions in samples in speech signal (int)
        samp_freq (int):            Sampling frequency

    Returns:
        (:obj:`numpy.array`): Array of targets - 1 = peaks with GCI associated or 0 = peaks without corrresponding GCIs

    """
    # Shift GCIs to the nearest peak
    gci_shifted = set()
    for p0 in gci.get_all_pmks(pm_type_incl={OnePm.type_V}):
        p0samp = p0.time*samp_freq
        pos = bisect_left(peaks, p0samp)
        if pos == 0:
            # Treat very first peak
            p1 = peaks[0]
        elif pos == len(peaks):
            # Treat very last peak
            p1 = peaks[-1]
        else:
            # Process inner peaks
            before, after = peaks[pos-1], peaks[pos]
            # Find the closest peak from the two around
            p1 = after if after-p0samp < p0samp-before else before
        logger.debug('{} shifted to peak time {}'.format(p0, p1/samp_freq))
        # Add GCI synced to (negative) peak
        gci_shifted.add(p1)

    # Set up targets - each peak is a candidate
    y = np.array([p in gci_shifted for p in peaks], dtype='int8')
    return y


def targets2gci(targets, peaks, samp_freq, new_peaks=None):
    """Shift targets to the nearest GCIs

        Args:
            targets (:obj:`numpy.array`):   Array of targets - 1 = corresponds to a GCI, 0 = does not correspond to any
                GCI
            peaks (:obj:`numpy.array`):     Array of (negative) peak positions in samples in speech signal (int)
            samp_freq (int):                Sampling frequency
            new_peaks (:obj:`numpy.array`): Array of (negative) peak positions in samples in speech signal (int)

        Returns:
            (:obj:`pm.Pm`): Pitch-mark object with the shifted GCIs

        """
    if new_peaks is None:
        pms = Pm()
        for samp in peaks[targets == 1]:
            logger.debug('Target sample {} corresponds to time {}'.format(samp, samp / samp_freq))
            pms.append(OnePm(samp / samp_freq, OnePm.type_V))
        return pms
    else:
        pms = Pm()
        for samp in peaks[targets == 1]:
            pos = bisect_left(new_peaks, samp-1)
            logger.debug('Target {} shifted to {}'.format(samp/samp_freq, new_peaks[pos-1]/samp_freq))
            pms.append(OnePm(new_peaks[pos-1]/samp_freq, OnePm.type_V))
        return pms


def prediction2samples(y, peaks, threshold=0.5):
    """Get GCI positions in samples.

    Args:
        y (array-like):             Predictions of peaks to be a GCI (1 = GCI, 0 = no GCI).
        peaks (:obj:`numpy.array`): Array of peak positions in samples (int).
        threshold (float):          Probability threshold.

    Returns:
        :obj:`numpy.array`: Array of GCI positions in samples (array of int with shape=(len(y),)).

    """
    gci = peaks[y > threshold]
    logger.debug('{} peaks out of {} correspond to GCIs after thresholding predictions on >{}'.format(len(gci), len(peaks), threshold))
    return gci


def sync_gci_to_samp_peak(y, peaks, samples, le, ri):
    """Synchronize GCIs with minimum signal peak in the given interval.

    Args:
        y (array-like):                 Predictions of peaks to be a GCI (1 = GCI, 0 = no GCI).
        peaks (:obj:`numpy.array`):     Array of peak positions in samples (int).
        samples (:obj:`numpy.array`):   Array of waveform samples (int).
        le (int):                       Position in samples to the left for syncing a predicted GCI with a sample peak
        ri (int):                       Position in samples to the right for syncing a predicted GCI with a sample peak

    Returns:
         :obj:`numpy.array`: Array of GCI positions (times) in samples (array of int).
    """
    warnings.warn('sync_gci_to_samp_peak() is deprecated, use sync_predictions_to_samp_peak instead',
                  DeprecationWarning)
    return sync_predictions_to_samp_peak(y, peaks, samples, le, ri)
    # # Start and end of the syncing interval (in samples)
    # logger.debug('Syncing {} predictions in interval {} - {}  (in samples)'.format(len(y), le, ri))
    # # Convert predictions to samples and sync them to minimum signal peak
    # return sync_time_to_samp_peak(prediction2samples(y, peaks), samples, le, ri)
    #
    # # sync = []
    # # n_samples = len(samples)
    # # for pred in prediction2samples(y, peaks):
    # #     beg = max(0, pred - le)
    # #     end = min(pred + ri, n_samples-1)
    # #     min_samp_idx = np.argmin(samples[beg:end+1]) + beg
    # #     sync.append(min_samp_idx)
    # #     logger.debug('GCI shifted in region ({:9.6f}, {:9.6f}): {:9.6f} --> {:9.6f}'.
    # #                  format(beg, end, pred, min_samp_idx))
    # # return np.array(sync)


def sync_predictions_to_samp_peak(y, peaks, samples, le, ri):
    """Synchronize GCIs with minimum signal peak in the given interval.

        Args:
            y (array-like):                 Predictions of peaks to be a GCI (1 = GCI, 0 = no GCI).
            peaks (:obj:`numpy.array`):     Array of peak positions in samples (int).
            samples (:obj:`numpy.array`):   Array of waveform samples (int).
            le (int):                       Position in samples to the left for syncing a predicted GCI with a sample peak
            ri (int):                       Position in samples to the right for syncing a predicted GCI with a sample peak

        Returns:
             :obj:`numpy.array`: Array of GCI positions (times) in samples (array of int).

        """
    # Convert predictions to samples and sync them to minimum signal peak
    return sync_time_to_samp_peak(prediction2samples(y, peaks), samples, le, ri)


def sync_time_to_samp_peak(marks, samples, le, ri):
    """Synchronize times with minimum signal peak in the given interval.

    Args:
        marks (:obj:`numpy.array`):   Times in samples.
        samples (:obj:`np.array`):    Array of waveform samples (int).
        le (int, :obj:`numpy.array`): Position (array of positions) in samples to the left for syncing the given time.
        ri (int, :obj:`numpy.array`): Position (array of positions) in samples to the left for syncing the given time.

    Warnings:
        If the intervals are given by vectors, their lengths must be rhe same as the length of time marks `marks`.

    Returns:
        :obj:`numpy.array`: Array of GCI positions (times) in samples (array of int).

    """
    sync = []
    n_sig = len(samples)

    # Prepare intervals to be always vectors
    if np.issubdtype(type(le), np.integer):
        le = le * np.ones(len(marks), dtype=int)
    else:
        if len(le) != len(marks):
            raise RuntimeError('Number of time marks and left contexts do not match')
    if np.issubdtype(type(ri), np.integer):
        ri = ri * np.ones(len(marks), dtype=int)
    else:
        if len(ri) != len(marks):
            raise RuntimeError('Number of time marks and right contexts do not match')

    logger.debug('# of samples in signal: {}'.format(n_sig))
    # Go through all time marks and sync them to the nearest negative peak given the interval
    for m, l, r in zip(marks, le, ri):
        # Prepare the interval
        beg, end = max(0, m-l), min(m+r, n_sig-1)
        logger.debug('Syncing {} (-{} +{}): ({} - {}) --> ({} - {})'.format(m, l, r, m-l, m+r, beg, end))
        # Find the negative peak within the given interval
        min_idx = np.argmin(samples[beg:end+1]) + beg
        sync.append(min_idx)
        logger.debug('{} --> {} ({} - {})'.format(m, min_idx, beg, end))
    return np.array(sync)

# def sync_gci_to_samp_peak2(y, peaks, samples, le, ri):
#     """Synchronize GCIs with minimum signal peak in the given interval
#
#     Args:
#         y (array-like):                 Predictions of peaks to be a GCI (1 = GCI, 0 = no GCI).
#         peaks (:obj:`numpy.array`):     Array of peak positions in samples (int).
#         samples (:obj:`numpy.array`):   Array of waveform samples (int).
#         le (float):                     Position in samples to the left for syncing a predicted GCI with a sample peak
#         ri (float):                     Position in samples to the right for syncing a predicted GCI with a sample
# peak
#
#     Returns:
#          :obj:`numpy.array`: Array of GCI positions (times) in samples (array of int).
#
#     """
#     # Start and end of the syncing interval (in samples)
#     logger.debug('Syncing {} predictions in interval {} - {}  (in samples)'.format(len(y), le, ri))
#     sync = []
#     n_samples = len(samples)
#     pms = Pm()
#     for samp in prediction2samples(y, peaks):
#         pms.append(OnePm(samp/16000., OnePm.type_V))
#     avg_f0 = pms.get_f0(Pm.bound_idx, 0, len(pms)-1)
#     avg_f0 = 150 if avg_f0 < 50 else avg_f0
#     logger.debug('Average F0 = {}'.format(avg_f0))
#     for idx, pm in enumerate(pms.get_all_pmks()):
#         idx1 = max(0, idx-2)
#         idx2 = min(len(pms)-1, idx+2)
#         logger.debug('PM indeces for F0: {} - {}'.format(idx1, idx2))
#         loc_f0 = pms.get_f0(Pm.bound_idx, idx1, idx2)
#         loc_t0 = 1 / avg_f0 if loc_f0 == Pm.invalid_f0 else 1 / loc_f0
#         logger.debug('local F0 = {} ({} - {})'.format(loc_f0, idx1, idx2))
#         pred = pm.time
#         beg = max(0, seconds2samples(pred - le*loc_t0, 16000))
#         end = min(seconds2samples(pred + ri*loc_t0, 16000), n_samples-1)
#         min_samp_idx = np.argmin(samples[beg:end+1]) + beg
#         sync.append(min_samp_idx)
#         logger.debug('GCI shifted in region ({}, {}): {} --> {}'.
#                      format(beg, end, seconds2samples(pred, 16000), min_samp_idx))
#     return np.array(sync)


def samples2seconds(samp, samp_freq):
    """Convert position in samples to seconds.

    Args:
        samp (:obj:`numpy.array`, int): Array of positions in samples (int) or a single position (int)
        samp_freq (int):                Sampling frequency

    Returns:
        :obj:`numpy.array` or float: Array of positions in seconds (float) or a single position in seconds (float)

    """
    return samp/samp_freq


def seconds2samples(t, samp_freq):
    """Convert position in seconds to position in samples.

    Args:
        t (:obj:`numpy.array`, float):  Array of positions in seconds (float) or a single position (float)
        samp_freq (int):                Sampling frequency

    Returns:
        :obj:`numpy.array` or int: Array of positions in samples (int) or a single position in samples (inr)

    """
    if isinstance(t, np.ndarray):
        t = np.rint(t*samp_freq).astype(int)
    else:
        t = int(round(t*samp_freq))
    return t


def sync_type(x, samp_freq, pms, f0, reg=0.75, polarity='-'):
    """Examine whether GCIs shall be synced to the right or to the left.

    Args:
        x (:obj:`numpy.array`):                     Array of waveform samples (int).
        samp_freq (int):                            Sampling frequency (Hz)
        pms (:obj:`pm.Pm`, :obj:`np.array`, list):  List of pitch-mark object or array-like object of pitch-mark times
        f0 (:obj:`numpy.array`):                    Array of F0 values for each pitch-mark (Hz)
        reg (float):                                Percentage of local T0 to examine around each pitch-mark
        polarity (str):                             Polarity of the input signal: '-' for findings negative peaks,
                                                    '+' for finding positive peaks

    Returns:
        shift (int):                    +1 for sync type 'plus' (to the right) or -1 for sync type 'minus' (to the left)
        n_le (int):                     The number of GCIs which shall be synced to the left
        n_ri (int):                     The number of GCIs which shall be synced to the right
        le_pos (:obj:`numpy.array`):    Array of min/max positions to the left
        ri_pos (:obj:`numpy.array`):    Array of min/max positions to the right

    """
    if len(pms) != len(f0):
        raise RuntimeError('Number of GCIs and F0s do not match')
    # Ensure that we are working with array of times in samples
    if isinstance(pms, Pm):
        pms = seconds2samples(np.array([p.time for p in pms]), samp_freq)
    elif isinstance(pms, (list, np.ndarray)):
        pms = seconds2samples(np.array(pms), samp_freq)
    # Init
    n_ri, n_le = 0, 0
    n_x = len(x)
    ri_pos, le_pos = np.empty(len(pms)), np.empty(len(pms))
    # Go through all pitch-marks
    for idx, (t, t0) in enumerate(zip(pms, seconds2samples(1/f0, samp_freq))):
        t0_max = int(t0*reg)
        logger.debug('GCI(#{:4d}): {:6d} T0={} T0max={}'.format(idx, t, t0, t0_max))
        # Right limit
        end = min(t+t0_max, n_x-1)
        logger.debug('Right limit for searching: {}'.format(end))
        # Find min/max to the right
        ri_idx = np.argmin(x[t:end]) + t if polarity == '-' else np.argmax(x[t:end]) + t
        ri = x[ri_idx]
        logger.debug('Right negative pulse: {} ({}) at {} of T0'.format(ri_idx, ri, (ri_idx-t)/t0))
        # Left limit
        beg = max(0, t-t0_max)
        logger.debug('Left limit for searching: {}'.format(beg))
        # Find min/max to the left
        le_idx = np.argmin(x[beg:t]) + beg if polarity == '-' else np.argmax(x[beg:t]) + beg
        le = x[le_idx]
        logger.debug('Left negative pulse: {} ({}) at {} of T0'.format(le_idx, le, (le_idx-beg)/t0))
        # Set counters
        if abs(ri) > abs(le):
            n_ri += 1
        else:
            n_le += 1
        ri_pos[idx], le_pos[idx] = (ri_idx-t)/t0, (t-le_idx)/t0
    logger.debug('Plus vs. minus: {} vs. {}'.format(n_ri, n_le))
    shift = 1 if n_ri > n_le else -1
    return shift, n_le, n_ri, le_pos, ri_pos
