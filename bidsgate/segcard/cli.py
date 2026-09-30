"""segcard NAME [--bidsgate scores_lesions.json] [--rescan summary.json ...] [--reference dice.tsv] --out card"""

import argparse
import json

from .core import build, render


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="segcard", description=__doc__)
    ap.add_argument("name")
    ap.add_argument("--bidsgate")
    ap.add_argument("--rescan", nargs="*", default=[])
    ap.add_argument("--reference")
    ap.add_argument("--out", required=True, help="writes OUT.json and OUT.html")
    a = ap.parse_args(argv)
    c = build(a.name, a.bidsgate, a.rescan, a.reference)
    with open(a.out + ".json", "w") as fh:
        json.dump(c, fh, indent=2)
    with open(a.out + ".html", "w") as fh:
        fh.write(render(c))
    print(a.out + ".html")
    return 0
