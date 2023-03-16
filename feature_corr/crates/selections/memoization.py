from loguru import logger
import pandas as pd


class Memoize:
    """Dimensionality reduction and visualisation"""

    def __init__(self) -> None:
        self.frame_memory = None

    def set_mem(self, frame: pd.DataFrame) -> tuple:
        """Memoization of the frame"""
        self.frame_memory = frame
        logger.info(f'Frame memory chached')
        return frame, None

    def get_mem(self, frame: pd.DataFrame) -> tuple:
        """Memoization of the frame"""
        logger.info(f'Frame memory retrieved')
        return self.frame_memory, None