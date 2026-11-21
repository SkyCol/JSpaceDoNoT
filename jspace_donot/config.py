"""Public configuration. Defaults preserve the supplied experimental settings."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    root: Path = Path('.')
    schedule: str = 'wait'
    every_n_tokens: int = 256
    wait_markers: tuple[str,...] = ('Wait',)
    min_stop_tokens: int = 256
    max_monitor_tokens: int | None = None
    calibration: int = 256
    test_limit: int | None = None
    max_new_tokens: int = 8192
    cache_mode: str = 'resume'  # resume | read_only | off
    cache_root: Path | None = None
    source_mode: str = 'auto'  # GSM8K: auto | historical | fresh
    output: Path | None = None
    lens_path: Path | None = None

    def validate(self):
        if self.dataset not in {'gsm8k','math500','raceh'}:raise ValueError('Unknown dataset')
        if self.schedule not in {'wait','token'}:raise ValueError('schedule must be wait or token')
        if self.cache_mode not in {'resume','read_only','off'}:raise ValueError('Invalid cache_mode')
        if self.source_mode not in {'auto','historical','fresh'}:raise ValueError('Invalid source_mode')
        if self.cache_mode=='off' and self.source_mode=='historical':raise ValueError('historical requires a persistent source cache')
        if self.calibration<4:raise ValueError('calibration must be >=4')
        if min(self.every_n_tokens,self.min_stop_tokens,self.max_new_tokens)<=0:raise ValueError('Token counts must be positive')
        if self.test_limit is not None and self.test_limit<=0:raise ValueError('test_limit must be positive')
        if self.max_monitor_tokens is not None and self.max_monitor_tokens<self.min_stop_tokens:raise ValueError('max_monitor_tokens below min_stop_tokens')
        if not self.wait_markers or not all(isinstance(m,str) and m.strip() for m in self.wait_markers):raise ValueError('Empty wait marker')
        return self
