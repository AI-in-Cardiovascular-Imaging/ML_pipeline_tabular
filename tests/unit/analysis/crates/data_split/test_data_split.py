import pandas as pd
from pytest import mark

df = pd.DataFrame(
    {
        'A': [0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        'B': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        'C': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        'D': [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        'E': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        'F': [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
        'G': [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        'H': [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
        'I': [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
        'J': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
    }
)


@mark.data_split
class DataSplitTests:
    @staticmethod
    @mark.parametrize('element', [df])
    def test_1(element, data_split):
        data_split.frame = element
        data_split.data_split()
        sel_train = data_split._frame_store['test']['selection_train']
        del sel_train['A']
        sel_train = set(sel_train.to_numpy().flatten())
        ver_train = data_split._frame_store['test']['verification_train']
        del ver_train['A']
        ver_train = set(ver_train.to_numpy().flatten())
        ver_test = data_split._frame_store['test']['verification_test']
        del ver_test['A']
        ver_test = set(ver_test.to_numpy().flatten())
        assert len(sel_train.intersection(ver_train)) == 0, 'Selection and validation train sets are intersecting'
        assert len(ver_train.intersection(ver_test)) == 0, 'Validation train and test sets are intersecting'
        assert len(sel_train.intersection(ver_test)) == 0, 'Selection and validation test sets are intersecting'
