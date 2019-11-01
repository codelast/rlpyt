import pyprind

from rlpyt.utils.logging import logger


class ProgBarCounter:
    """
    该类负责更新训练过程中的进度条更新显示。
    """

    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def update(self, current_count):
        """
        更新进度条显示。
        """
        if not logger.get_log_tabular_only():
            self.cur_count = current_count
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        """
        停止更新进度条。
        """
        if self.pbar is not None and self.pbar.active:
            self.pbar.stop()
