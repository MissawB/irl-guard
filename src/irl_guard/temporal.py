"""Hystérésis de la pause : entrée immédiate, sortie après K analyses calmes."""

from __future__ import annotations


class Hysteresis:
    def __init__(self, haut: float, bas: float, k: int):
        self.haut, self.bas, self.k = haut, bas, k
        self.en_pause = False
        self._calmes = 0

    def update(self, score: float) -> bool:
        if score >= self.haut:
            self.en_pause, self._calmes = True, 0
        elif self.en_pause:
            if score < self.bas:
                self._calmes += 1
                if self._calmes >= self.k:
                    self.en_pause, self._calmes = False, 0
            else:
                self._calmes = 0
        return self.en_pause
