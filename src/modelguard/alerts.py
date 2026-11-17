"""Alert engine: a monitor alerts only after N consecutive breaching batches."""
from dataclasses import dataclass

from modelguard.monitors import MonitorResult


@dataclass(frozen=True)
class Alert:
    monitor: str
    feature: str
    batch_index: int
    consecutive: int
    reason: str


class AlertEngine:
    def __init__(self, consecutive: int):
        if consecutive < 1:
            raise ValueError("consecutive must be >= 1")
        self.n = consecutive
        self.streaks: dict[tuple[str, str], int] = {}

    def update(self, batch_index: int, results: list[MonitorResult]) -> list[Alert]:
        """Feed one batch of results. One alert fires per streak, when it reaches N; a clear batch resets it.
        A monitor absent from `results` (e.g. no labels yet) counts as clear."""
        flagged = {(r.monitor, r.feature): r for r in results if r.flagged}
        self.streaks = {k: v for k, v in self.streaks.items() if k in flagged}
        out = []
        for key, r in flagged.items():
            self.streaks[key] = self.streaks.get(key, 0) + 1
            if self.streaks[key] == self.n:
                p = f", p={r.p_value:.3g}" if r.p_value is not None else ""
                out.append(Alert(*key, batch_index, self.n,
                                 f"{r.monitor} on {r.feature} breached its threshold for {self.n} consecutive "
                                 f"batches (statistic={r.statistic:.4g}{p})"))
        return out
