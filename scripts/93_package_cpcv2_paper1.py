#!/usr/bin/env python3
"""Package the editable arXiv source and verified manuscript artifacts.

Does not compile, publish, or include private gauge data. Re-run after filling
any pending manuscript slot. Uses only the Python standard library.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def manuscript_assets(source):
    """Find literal graphics and filled optional slots in the current source.

    Do not use a figure list from an older draft. Missing guarded assets remain
    pending, but a missing unconditional figure makes the package incomplete.
    """
    text = re.sub(r"(?<!\\)%[^\n]*", "", source.read_text())
    guards = set(re.findall(r"\\IfFileExists\{([^}]+)\}",text))
    files, pending = [], []
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",text):
        if "#" in name:
            continue  # The optionalfigure macro is resolved from its calls below.
        candidates = [source.parent/name, source.parent/"figures"/name]
        path = next((p for p in candidates if p.is_file()),None)
        if path is None:
            if name in guards or "figures/"+name in guards:
                pending.append(name)
                continue
            raise FileNotFoundError(f"required manuscript figure missing: {name}")
        files.append(path)
    for macro,suffix in (("optionalfigure",".pdf"),("optionaltable",".tex")):
        for name in re.findall(r"\\"+macro+r"\{([^}]+)\}",text):
            path = source.parent/"additional"/(name+suffix)
            if path.is_file():files.append(path)
            else:pending.append(str(path.relative_to(source.parent)))
    # Absolute or escaping paths would produce a nonportable source bundle.
    for path in files:
        path.resolve().relative_to(source.parent.resolve())
    return list(dict.fromkeys(files)),sorted(set(pending))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=ROOT / "manuscript")
    parser.add_argument("--output", type=Path, default=ROOT / "manuscript/SURMA_Flow_CPCv2_arxiv_source.zip")
    args = parser.parse_args()
    source = args.manuscript / "BDhighresDA_arxiv.tex"
    figure_manifest = args.manuscript / "figures/figure_manifest.json"
    assets,pending = manuscript_assets(source)
    files = [source]+assets
    # Metadata is retained for provenance; it does not select figure files.
    for path in (figure_manifest,args.manuscript/"figures/restructured_numbers.json"):
        if path.is_file():files.append(path)
    completion = args.manuscript / "additional/completion_manifest.json"
    if completion.is_file():
        manifest = json.loads(completion.read_text())
        files.append(completion)
        for record in manifest["outputs"]:
            path = completion.parent / record["path"]
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"missing or changed generated artifact: {path}")
            if path.suffix in (".tex", ".pdf"):
                files.append(path)
    files = list(dict.fromkeys(files))
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(args.manuscript))
        archive.writestr("README.txt", "SURMA-Flow CPCv2 model-paper source.\n"
            "Compile BDhighresDA_arxiv.tex with figures/ and additional/ beside it.\n"
            "Use pdflatex twice or a compatible whole-project TeX compiler.\n"
            "Standard article class; inline bibliography; no BibTeX dependency.\n"
            "Pending boxes mark unavailable evidence, not estimated results.\n"
            "Graphics are inventoried from the current TeX source, not an older figure manifest.\n"
            "Confirm author metadata and evidence provenance before submission.\n")
        archive.writestr("package_manifest.json",json.dumps({
            "inventory_source":"current manuscript TeX literal graphics and available optional slots",
            "files":[{"path":str(p.relative_to(args.manuscript)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
            "pending_assets":pending},indent=2)+"\n")
        for name in ("PAPER1_PREPARATION_NOTES.md",):
            if (args.manuscript / name).is_file():
                archive.write(args.manuscript / name, name)
        archive.write(ROOT / "docs/PAPER1_COMPLETION.md", "PAPER1_COMPLETION.md")
        archive.write(ROOT / "docs/PAPER1_BANGLADESH.md", "PAPER1_BANGLADESH.md")
        archive.write(ROOT / "docs/PAPER1_UPDATED_EVIDENCE.md", "PAPER1_UPDATED_EVIDENCE.md")
        archive.write(ROOT / "docs/PAPER1_REMAINING_EVIDENCE.md", "PAPER1_REMAINING_EVIDENCE.md")
        for path in (ROOT / "configs/geography").glob("geoBoundaries-BGD-ADM0*.json"):
            archive.write(path, "geography/" + path.name)
        boundary = ROOT / "configs/geography/geoBoundaries-BGD-ADM0.geojson"
        archive.write(boundary, "geography/" + boundary.name)
    print(f"[paper1-package] {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
