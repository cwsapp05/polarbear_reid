"""Stage 0: download PolarBearVidID, build the frame catalogue and the frame-stride sample.

Outputs: work/catalogue.csv (all frames), work/sample.csv (every k-th frame per sequence).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile

from common import base_argparser, build_catalogue, get_logger, load_config, work_paths

log = get_logger("prepare")


def md5sum(path, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest) -> None:
    def hook(blocks, bsize, total):
        if total > 0 and blocks % 200 == 0:
            sys.stdout.write(f"\r  {min(blocks * bsize / total, 1):6.1%} of {total / 1e9:.2f} GB")
            sys.stdout.flush()
    log.info("Downloading %s", url)
    urllib.request.urlretrieve(url, dest, hook)
    print()


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--skip-download", action="store_true", help="dataset is already extracted")
    args = ap.parse_args()
    cfg = load_config(args.config)
    wp = work_paths(cfg)
    dl = cfg["paths"]["download_dir"]
    dl.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        zpath = dl / "PolarBearVidID.zip"
        if not zpath.exists():
            download(cfg["dataset"]["url"], zpath)
        got = md5sum(zpath)
        if got != cfg["dataset"]["md5"]:
            raise SystemExit(f"MD5 mismatch for {zpath}: {got} (expected {cfg['dataset']['md5']}). "
                             "Delete the file and re-run.")
        log.info("MD5 OK")
        marker = dl / ".extracted"
        if not marker.exists():
            log.info("Extracting...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dl)
            marker.touch()

    cat = build_catalogue(cfg)
    cat.to_csv(wp["catalogue"], index=False)

    # ---- sanity checks against the published counts
    exp = cfg["dataset"]["expected"]
    n_id, n_seq, n_img = cat["id"].nunique(), cat["sequence"].nunique(), len(cat)
    log.info("Catalogue: %d identities, %d sequences, %d images (paper: %d / %d / %d)",
             n_id, n_seq, n_img, exp["identities"], exp["sequences"], exp["images"])
    log.info("Example file names: %s", ", ".join(cat["file"].head(3)))
    if abs(n_seq - exp["sequences"]) > 0.05 * exp["sequences"]:
        log.warning("Sequence count is far from the paper's. The file-name -> sequence rule is probably "
                    "wrong for this release; set dataset.sequence_regex in config.yaml. This matters: "
                    "same-sequence pairs are excluded from evaluation.")
    if cat["zoo"].eq("unknown").all():
        log.warning("animal_db.csv has no 'zoo' column; the within-zoo analysis will be skipped.")
    else:
        log.info("Bears per zoo:\n%s", cat.groupby("zoo")["identity"].nunique().to_string())

    k = int(cfg["dataset"]["frame_stride"])
    sample = cat[cat["frame_idx"] % k == 0].reset_index(drop=True)
    sample.to_csv(wp["sample"], index=False)
    log.info("Sample (every %d-th frame): %d images -> %s", k, len(sample), wp["sample"])


if __name__ == "__main__":
    main()
