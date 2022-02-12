#!/usr/bin/env python

import os

import cv2
import matplotlib.pyplot as plt
import numpy.random as rand
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

n_classes = 3
n_samples = 150
max_n_dimensions = 150


def plot_LDA(n_dimensions=3):
    # Test and train set creation
    # X is feature set, y is label set
    X = df.iloc[:, :n_dimensions].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Linear Discriminant Analysis
    num_components = 2
    lda = LDA(n_components=num_components)
    x = lda.fit_transform(X_train, y_train)
    x = pd.DataFrame(x, columns=[f'LD{i+1}' for i in range(num_components)])

    # Plot
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x['LD1'], x['LD2'], c=y_train, cmap='rainbow')
    plt.title(f'LDA on Random Data ({n_dimensions} dimensions)')
    plt.xlabel('LD1')
    plt.ylabel('LD2')

    plt.savefig(f'.images/dimensions_{n_dimensions:03}')
    plt.close()


if __name__ == '__main__':
    # === Create data
    df = pd.DataFrame(rand.random((n_samples, max_n_dimensions)), columns=[f'Dim_{i}' for i in range(max_n_dimensions)])
    _class = sorted([i % n_classes for i in range(n_samples)])
    df['Class'] = _class

    # === Create LDA plots
    dimensions = [i for i in range(3, max_n_dimensions)]
    for i in dimensions:
        plot_LDA(i)

    # === Create video (https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python)
    image_folder = '.images'
    video_name = 'LDA-dimensionality.avi'

    images = sorted([img for img in os.listdir(image_folder) if img.endswith('.png')])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 5, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
