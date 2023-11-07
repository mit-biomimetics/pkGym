import time


class TimeKeeper:
    def __init__(self):
        self._tic_time = {}
        self._toc_time = {}

    def tic(self, category="default"):
        self._tic_time.update({category: time.time()})
        if category not in self._toc_time.keys():
            self._toc_time.update({category: -1})
        return None

    def toc(self, category="default"):
        time_elapsed = time.time() - self._tic_time[category]
        self._toc_time.update({category: time_elapsed})
        return None

    def get_time(self, category="default"):
        return self._toc_time[category]
