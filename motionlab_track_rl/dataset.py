from __future__ import annotations

"""Reference dataset utilities.

This supports the *pretrained* tracking workflow:

* **Training**: learn a single tracking policy that can follow many references, sampled from a
  dataset of athlete videos (converted to .npz references).
* **Inference**: accept a brand-new user video, extract its reference on the fly, and run the
  pretrained tracking policy (no per-user RL training required).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .reference import ReferenceSequence, load_reference_npz


@dataclass
class ReferenceDataset:
    """Lazy-loading dataset of reference .npz files."""

    root: Path
    files: List[Path]
    _cache: Optional[Dict[str, ReferenceSequence]] = None

    @classmethod
    def from_dir(cls, root: str, pattern: str = "*.npz", cache: bool = True) -> "ReferenceDataset":
        root_p = Path(root)
        if not root_p.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        files = sorted(root_p.rglob(pattern))
        if len(files) == 0:
            raise FileNotFoundError(f"No reference npz files found in {root} (pattern={pattern})")
        return cls(root=root_p, files=files, _cache={} if cache else None)

    def __len__(self) -> int:
        return len(self.files)

    def get(self, idx: int) -> ReferenceSequence:
        p = self.files[int(idx) % len(self.files)]
        key = str(p)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        ref = load_reference_npz(str(p))
        if self._cache is not None:
            self._cache[key] = ref
        return ref

    def sample(self, rng: np.random.RandomState) -> ReferenceSequence:
        """Sample a random reference sequence."""
        idx = int(rng.randint(0, len(self.files)))
        return self.get(idx)
