"""Smoke tests for the visualisation module (figures are written to disk)."""

from __future__ import annotations

from breachlab.config import Config
from breachlab.evaluation import run_evaluation
from breachlab.viz import save_all_figures


def test_save_all_figures_writes_pngs(tmp_path) -> None:
    cfg = Config()
    cfg.sim.duration_s = 300
    cfg.sim.n_centrifuges = 3
    cfg.detector.ae_epochs = 2       # keep it fast
    res = run_evaluation(cfg)

    paths = save_all_figures(res, tmp_path)
    # 2 variants x (real-vs-reported + scores) + 1 F1 chart = 5 figures.
    assert len(paths) == 5
    for p in paths:
        assert p.exists()
        assert p.suffix == ".png"
        assert p.stat().st_size > 0
