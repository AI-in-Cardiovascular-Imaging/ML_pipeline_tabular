import pandas as pd


class Memoize:
    """Memoization of the frame"""

    def __init__(self) -> None:
        self.frame_memory = None

    def set_memory(self, frame: pd.DataFrame) -> tuple:
        """Cache the frame"""
        self.frame_memory = frame
        return frame, None

    def get_memory(self) -> tuple:
        """Retrieve the frame"""
        if self.frame_memory is None:
            raise ValueError('No frame available in memory')
        return self.frame_memory, None
