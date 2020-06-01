# %%
import gudhi as gd
import gudhi.representations.vector_methods as grvm

import wfdb 
from wfdb import processing
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import scipy as scp
import argparse
from tqdm import tqdm



# %%
def quasi_attractorize(series, step):
    n = series.shape[0]

    z = []
    for i in range(n - step + 1):
        z.append(series[i:i + step].reshape(1, step)[0])

    return np.array(z)


# %%
def plot_attractor(attractor):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter(xs=attractor[:, 0],
            ys=attractor[:, 1],
            zs=attractor[:, 2])
    plt.title(f'{len(attractor)}')
    plt.show()

# %%
beat_annotations = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']

non_beat_annotations = ['[', '!', ']', 'x', '(', ')', 'p', 't', 'u', '`', "'", '^', '|', '~', 's', 'T', '*', 'D', '=', '"', '@']
ignore_annotations = ['Q', '?']

classes_mapping = {
    'N': 0,
    '/': 0,
    'f': 0,
    'F': 1,
    'L': 1, 
    'R': 1,
    'B': 1,
    'A': 1,
    'a': 1, 
    'J': 1,
    'S': 1,
    'V': 1,
    'r': 1,
    'e': 1,
    'j': 1,
    'n': 1,
    'E': 1, 
    'Q': -1,
    '?': -1
}


# %%
def extract_attractors(signal, annotations, event_indexes, step):
    n = event_indexes.shape[0]

    attractors = []
    attractor_annotations = []
    for i in range(n - step + 1):
        idx1 = event_indexes[i]
        idx2 = event_indexes[i+step-1] + 1 
        
        extracted_signal = signal[idx1:idx2]
        
        attractor = quasi_attractorize(extracted_signal, step=3)

        ann = 1 if 1 in annotations[i:i+step] else 0

        attractors.append(attractor)
        attractor_annotations.append(ann)

    return attractors, attractor_annotations


# %%
def filter_signal(signal, cutoff_freqs, fs, numtaps):
    filter_taps = scp.signal.firwin(numtaps=numtaps, fs=fs, cutoff=cutoff_freqs)
    return scp.signal.lfilter(filter_taps, 1.0, x=signal)


# %%
def extract_numpy_from_diag(diagram):
    tuples = [x[1] for x in diagram]
    xs = np.array([d[0] for d in tuples])
    ys = np.array([d[1] for d in tuples])

    return np.array([xs, ys]).T


# %%
def preprocess(record, annotation, new_fs, numtaps, cutoff_freqs):
    # Resampling to 200 Hz
    zero_channel_signal, resampled_ann = processing.resample_singlechan(record.p_signal[:, 0], annotation, record.fs, new_fs)

    # FIR filtering 
    # TODO how to choose numtaps???
    filtered = filter_signal(zero_channel_signal, cutoff_freqs, record.fs, numtaps)

    # Normalizing signal to 0, 1
    preprocessed_signal = processing.normalize_bound(filtered, lb=0, ub=1)

    return preprocessed_signal, resampled_ann


# %%
def read_samples(dirpath, limit=None):
    sample_names = []
    with open(f'{dirpath}/RECORDS', 'r') as f:
        try:
            while line := f.readline():
                sample_names.append(line[:-1]) # trim \n
        except IOError:
            print(f'Error while reading {dirpath}/RECORDS')
    
    records = []
    annotations = []
    for i, sample in enumerate(sample_names):
        # TODO remove limited reading of the samples
        current_annotations = wfdb.rdann(f'{dirpath}/{sample}', 'atr', sampfrom=0, sampto=10000)
        
        annotations.append(current_annotations)
        records.append(wfdb.rdrecord(f'{dirpath}/{sample}', sampfrom=0, sampto=10000))

        if i == limit-1:
            break

    return records, annotations


# %%
def extract_betti_curve(points, min_range, max_range, resoultion):
    gudhi_complex = gd.RipsComplex(points=points)
    simplex_tree = gudhi_complex.create_simplex_tree(max_dimension=2)

    persistence = simplex_tree.persistence()

    betti_curve = grvm.BettiCurve(sample_range=[min_range, max_range], resolution=resoultion)
    curve = betti_curve(extract_numpy_from_diag(persistence))

    return curve


# %%
def get_betti_curves(attractors, betti_curve_length):
    return np.array([extract_betti_curve(attractor, 0.01, 1, betti_curve_length) for attractor in attractors])


# %%
def preprocess_flow(samples, annotations, betti_curve_length):
    curves = []
    anns = []

    for i, record in enumerate(tqdm(samples)):
        prep_signal, resampled_annotation = preprocess(record, annotations[i], new_fs=200, numtaps=21, cutoff_freqs=[0.5, 50])                                 

        mapped_ann = list(
            map(
                lambda x : classes_mapping[x] if x in classes_mapping else 2, 
                resampled_annotation.symbol
                )
        )

        attractors, attractor_anns = extract_attractors(prep_signal, mapped_ann[1:], resampled_annotation.sample[1:], 3)

        betti_curves = get_betti_curves(attractors, betti_curve_length)

        curves.append(betti_curves)
        anns.append(attractor_anns)

    matrix = np.vstack(curves)
    stacked_annotations = np.hstack(anns)

    df = pd.DataFrame(matrix, columns=range(betti_curve_length))
       
    df['class'] = stacked_annotations

    return df


# %%
def main():
    # TODO remove limited reading of the samples
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Relative path to the data folder')
    parser.add_argument('--save_path', help='Relative path to the data folder where preprocessed data should be saved')
    parser.add_argument('--limit', type=int, help='Number of samples which should be preprocessed')
    parser.add_argument('--betti_length', type=int, help='Length of the generated betti curves')
    args = parser.parse_args()

    arrhythmia_records, arrhythmia_anns = read_samples(args.data_path, args.limit)

    df = preprocess_flow(arrhythmia_records, arrhythmia_anns, args.betti_length)
    df.to_csv(args.save_path, index=False)


# %%
if __name__ == '__main__':
    main()
