"""Thin Python wrapper around the BearID command-line tools (bearface, bearchip, bearembed).

BearID (github.com/bearid/bearid-dlib) is a set of dlib C++ programs that talk to each other
through imglab-style XML files. This module writes those XML files, runs the binaries, and
parses their output back into Python.
"""
from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

LANDMARKS = ("head_top", "lear", "leye", "nose", "rear", "reye")


def binary(cfg: dict, name: str) -> Path:
    p = cfg["paths"]["bearid_bin_dir"] / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Build BearID with ./setup_bearid.sh "
                                "or point paths.bearid_bin_dir at the build folder.")
    return p


def network(cfg: dict, key: str) -> Path:
    p = cfg["paths"]["bearid_models_dir"] / cfg["bearid"][key]
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Clone github.com/hypraptive/bearid-models "
                                "(setup_bearid.sh does this).")
    return p


def run(cmd: list, cwd: Path, log_path: Path) -> None:
    with open(log_path, "w") as lf:
        r = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed (exit {r.returncode}); see {log_path}")


# --------------------------------------------------------------------------- XML I/O
def _write(tree_root: ET.Element, path: Path) -> None:
    ET.indent(tree_root)
    ET.ElementTree(tree_root).write(path, encoding="ISO-8859-1", xml_declaration=True)


def write_image_list(paths: list[str], xml_path: Path) -> None:
    """imglab-style dataset with no boxes: the input format for `bearface --infer`."""
    root = ET.Element("dataset")
    ET.SubElement(root, "name").text = "polarbear"
    images = ET.SubElement(root, "images")
    for p in paths:
        ET.SubElement(images, "image", file=str(p))
    _write(root, xml_path)


def parse_faces(xml_path: Path) -> dict[str, list[dict]]:
    """bearface output -> {image file: [{'left','top','width','height','parts':{name:(x,y)}}]}"""
    out = {}
    for img in ET.parse(xml_path).getroot().iter("image"):
        faces = []
        for b in img.findall("box"):
            parts = {p.get("name"): (float(p.get("x")), float(p.get("y"))) for p in b.findall("part")}
            faces.append({k: float(b.get(k)) for k in ("left", "top", "width", "height")} | {"parts": parts})
        out[img.get("file")] = faces
    return out


def write_faces(faces: dict[str, dict], labels: dict[str, str], xml_path: Path) -> None:
    """One face per image, in the format bearchip reads."""
    root = ET.Element("dataset")
    ET.SubElement(root, "name").text = "polarbear faces"
    images = ET.SubElement(root, "images")
    for f, face in faces.items():
        im = ET.SubElement(images, "image", file=str(f))
        box = ET.SubElement(im, "box", **{k: str(int(round(face[k]))) for k in ("top", "left", "width", "height")})
        ET.SubElement(box, "label").text = labels[f]
        for name, (x, y) in face["parts"].items():
            ET.SubElement(box, "part", name=name, x=str(int(round(x))), y=str(int(round(y))))
    _write(root, xml_path)


def parse_chips(xml_path: Path) -> dict[str, str]:
    """bearchip output -> {chip jpg path: source image path}"""
    out = {}
    for chip in ET.parse(xml_path).getroot().iter("chip"):
        src = chip.find("source")
        if src is not None:
            out[chip.get("file")] = src.get("file")
    return out


def parse_embeddings(xml_path: Path) -> dict[str, np.ndarray]:
    """bearembed output -> {chip jpg path: 128-d embedding}"""
    out = {}
    for e in ET.parse(xml_path).getroot().iter("embedding"):
        vals = e.findtext("embed_val", default="").split()
        out[e.findtext("chip_source")] = np.asarray(vals, dtype=np.float32)
    return out


# --------------------------------------------------------------------------- face detection
def detect_faces(cfg: dict, image_paths: list[str], out_dir: Path, n_workers: int = 1,
                 logger=None) -> dict[str, list[dict]]:
    """Run bearface on `image_paths` in n_workers parallel shards; return parsed faces."""
    exe, net = binary(cfg, "bearface"), network(cfg, "face_network")
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = [s for s in np.array_split(np.asarray(image_paths, dtype=object), max(1, n_workers)) if len(s)]

    def job(i_shard):
        i, shard = i_shard
        xml = out_dir / f"shard_{i:02d}.xml"
        faces_xml = out_dir / f"shard_{i:02d}_faces.xml"
        if not faces_xml.exists():  # resume support
            write_image_list(list(shard), xml)
            run([exe, "--infer", net, xml], cwd=out_dir, log_path=out_dir / f"shard_{i:02d}_bearface.log")
        if logger:
            logger.info("bearface shard %d/%d done (%d images)", i + 1, len(shards), len(shard))
        return parse_faces(faces_xml)

    faces = {}
    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as ex:
        for part in ex.map(job, enumerate(shards)):
            faces.update(part)
    return faces


def best_face(faces: list[dict], margin: float) -> tuple[dict | None, bool, float]:
    """Pick the largest face; check that all 6 landmarks lie inside the (expanded) face box.

    Returns (face, landmarks_ok, eye_distance / face_width).
    """
    if not faces:
        return None, False, np.nan
    f = max(faces, key=lambda b: b["width"] * b["height"])
    mx, my = margin * f["width"], margin * f["height"]
    inside = all(
        name in f["parts"]
        and f["left"] - mx <= f["parts"][name][0] <= f["left"] + f["width"] + mx
        and f["top"] - my <= f["parts"][name][1] <= f["top"] + f["height"] + my
        for name in LANDMARKS)
    eye = np.nan
    if "leye" in f["parts"] and "reye" in f["parts"]:
        eye = float(np.hypot(*np.subtract(f["parts"]["leye"], f["parts"]["reye"])) / max(f["width"], 1))
    return f, inside, eye
