import numpy as np
from minepy import MINE
from sklearn.feature_selection import SelectPercentile


def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5


def mic_select_features_helper(X, Y):
    def get_mic_results(x):
        return mic(x, Y)
    return tuple(map(tuple, np.array(list(map(get_mic_results, X.T))).T))


def mic_select_features(covid_features, covid_labels, percentile):
    # cannot serialize
    # mic_model = SelectPercentile(
    #     lambda X, Y: tuple(map(tuple, np.array(list(map(lambda x: mic(x, Y), X.T))).T)), percentile=percentile
    # )
    mic_model = SelectPercentile(
        mic_select_features_helper, percentile=percentile
    )
    mic_model.fit(covid_features, covid_labels)
    return mic_model


def manual_select_features(indices):
    class ManualSelection:
        def __init__(self, indices):
            self.indices = indices

        def transform(self, features):
            return features[:, self.indices]

    return ManualSelection(indices)
