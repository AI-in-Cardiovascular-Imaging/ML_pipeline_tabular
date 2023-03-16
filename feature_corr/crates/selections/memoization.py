from loguru import logger
import pandas as pd


class Memoize:
    """Memoization of the frame"""

    def __init__(self) -> None:
        self.frame_memory = None

    def set_mem(self, frame: pd.DataFrame) -> tuple:
        """Cache the frame"""
        self.frame_memory = frame
        logger.info(f'Frame cached')
        return frame, None

    def get_mem(self, frame: pd.DataFrame) -> tuple:
        """Retrieve the frame"""
        logger.info(f'Frame retrieved')
        return self.frame_memory, None