from pytest import mark
import pandas as pd

df = pd.DataFrame({'A': [0, 0, 0], 'B': [1, 2, 1], 'C': [-1, 0, 1], 'D': [1, 2, 3]})


@mark.normaliser
class NormaliserTests:
    @staticmethod
    @mark.parametrize('element', [df])
    def test_l1_norm(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [0.25, 0.50, 0.25], 'C': [-0.5, 0, 0.5], 'D': [1, 2, 3]})
        result = normaliser.l1_norm(element)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_l2_norm(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame(
            {
                'A': [0.0, 0.0, 0.0],
                'B': [0.408248, 0.816497, 0.408248],
                'C': [-0.707107, 0.000000, 0.707107],
                'D': [1, 2, 3],
            }
        )
        result = normaliser.l2_norm(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_z_score_norm(element, normaliser):
        normaliser.target_label = 'D'

        expected = pd.DataFrame(
            {
                'A': [0.0, 0.0, 0.0],
                'B': [-0.707107, 1.414214, -0.707107],
                'C': [-1.224745, 0.000000, 1.224745],
                'D': [1, 2, 3],
            }
        )
        result = normaliser.z_score_norm(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_min_max_scale(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [0.0, 1.0, 0.0], 'C': [0.0, 0.5, 1.0], 'D': [1, 2, 3]})
        result = normaliser.min_max_scale(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_max_abs_scale(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [0.5, 1.0, 0.5], 'C': [-1.0, 0.0, 1.0], 'D': [1, 2, 3]})
        result = normaliser.max_abs_scale(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_robust_scale(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [0.0, 2.0, 0.0], 'C': [-1.0, 0.0, 1.0], 'D': [1, 2, 3]})
        result = normaliser.robust_scale(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_quantile_transform(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame({'A': [0.0, 0.0, 0.0], 'B': [0.0, 1.0, 0.0], 'C': [0.0, 0.5, 1.0], 'D': [1, 2, 3]})
        result = normaliser.quantile_transform(element).round(6)
        assert result.equals(expected) is True

    @staticmethod
    @mark.parametrize('element', [df])
    def test_power_transform(element, normaliser):
        normaliser.target_label = 'D'
        expected = pd.DataFrame(
            {
                'A': [0.0, 0.0, 0.0],
                'B': [-0.707107, 1.414214, -0.707107],
                'C': [-1.224745, 0.000000, 1.224745],
                'D': [1, 2, 3],
            }
        )
        result = normaliser.power_transform(element).round(6)
        assert result.equals(expected) is True
