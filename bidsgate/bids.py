"""Minimal BIDS walking: enough to find anatomical images and write derivatives."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import __version__

ENTITY = re.compile(r"(sub-[A-Za-z0-9]+)(?:_(ses-[A-Za-z0-9]+))?.*_(T1w|FLAIR|T2w)\.nii(\.gz)?$")


@dataclass(frozen=True)
class Anat:
    subject: str
    session: str | None
    suffix: str
    path: Path

    @property
    def stem(self) -> str:
        return self.path.name.split(".")[0]

    @property
    def base(self) -> str:
        return f"{self.subject}_{self.session}" if self.session else self.subject


def find_anat(root: Path, suffix: str = "T1w", subjects: list[str] | None = None) -> list[Anat]:
    out = []
    for p in sorted(root.glob("sub-*/**/anat/*.nii*")):
        m = ENTITY.match(p.name)
        if not m or m.group(3) != suffix:
            continue
        sub, ses = m.group(1), m.group(2)
        if subjects and sub not in subjects and sub.removeprefix("sub-") not in subjects:
            continue
        out.append(Anat(sub, ses, suffix, p))
    return out


def sibling(anat: Anat, suffix: str) -> Path | None:
    """The same subject/session's image of another suffix, if present."""
    cand = anat.path.with_name(anat.path.name.replace(f"_{anat.suffix}.", f"_{suffix}."))
    return cand if cand.exists() else None


def derivative_path(out_root: Path, anat: Anat, suffix: str, desc: str | None = None, ext: str = ".nii.gz") -> Path:
    parts = [anat.subject]
    if anat.session:
        parts.append(anat.session)
    d = out_root / anat.subject / (anat.session or "") / "anat"
    d.mkdir(parents=True, exist_ok=True)
    name = "_".join(parts + ([f"desc-{desc}"] if desc else []) + [suffix]) + ext
    return d / name


def write_dataset_description(out_root: Path, name: str, source_root: Path, kind: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    desc = {
        "Name": name,
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "bidsgate", "Version": __version__, "Description": f"synthetic {kind} injection with known truth"}],
        "SourceDatasets": [{"URL": str(source_root)}],
    }
    with open(out_root / "dataset_description.json", "w") as fh:
        json.dump(desc, fh, indent=2)


def copy_json_sidecar(src_nii: Path, dst_nii: Path, extra: dict) -> None:
    """Carry the acquisition sidecar over and add what was done."""
    src = src_nii.with_suffix("").with_suffix(".json") if src_nii.name.endswith(".nii.gz") else src_nii.with_suffix(".json")
    meta = {}
    if src.exists():
        try:
            with open(src) as fh:
                meta = json.load(fh)
        except json.JSONDecodeError:
            meta = {}
    meta.update(extra)
    dst = Path(str(dst_nii).replace(".nii.gz", ".json").replace(".nii", ".json"))
    with open(dst, "w") as fh:
        json.dump(meta, fh, indent=2)
