"""Export a generated report bundle into the spiral-web site repository.

This script copies the static artifacts produced by ``generate_report.py`` into a
spiral-web asset directory so the report viewer can be embedded from
``_research/evaluation.md`` without additional build steps.
"""

import argparse
import shutil
from pathlib import Path

DEFAULT_TARGET_SUBDIR = "assets/evaluation/report/latest"
REQUIRED_REPORT_FILES = ("summary.json",)
UI_ASSET_FILES = ("viewer.js", "styles.css", "index.html")


def _ensure_report_assets(report_dir: Path) -> None:
    """Validate the generated report directory contains required assets.

    Parameters
    ----------
    report_dir
        Directory produced by ``generate_report.py``.

    Returns
    -------
    None
    """
    missing = [
        filename
        for filename in REQUIRED_REPORT_FILES
        if not (report_dir / filename).is_file()
    ]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing required report files in '{report_dir}': {missing_list}"
        )


def _copy_file(src: Path, dst: Path) -> None:
    """Copy one file and ensure its parent directory exists.

    Parameters
    ----------
    src
        Source file path.
    dst
        Destination file path.

    Returns
    -------
    None
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def export_report_bundle(
    report_dir: Path,
    spiral_web_dir: Path,
    target_subdir: str,
    clean_target: bool,
    include_figures: bool,
) -> Path:
    """Copy report assets from ``llm-delusions-evals`` into ``spiral-web``.

    Parameters
    ----------
    report_dir
        Source generated report directory.
    spiral_web_dir
        Root directory of the spiral-web repository.
    target_subdir
        Destination path relative to ``spiral_web_dir``.
    clean_target
        Whether to delete the destination directory before copying.
    include_figures
        Whether to copy the optional ``figures/`` directory.

    Returns
    -------
    Path
        Absolute path to the destination directory.
    """
    report_dir = report_dir.resolve()
    spiral_web_dir = spiral_web_dir.resolve()
    target_dir = (spiral_web_dir / target_subdir).resolve()
    ui_assets_dir = Path(__file__).resolve().parent / "report_assets"

    if not report_dir.is_dir():
        raise NotADirectoryError(f"Report directory not found: '{report_dir}'")

    if not spiral_web_dir.is_dir():
        raise NotADirectoryError(f"Spiral-web directory not found: '{spiral_web_dir}'")

    _ensure_report_assets(report_dir)

    if clean_target and target_dir.exists():
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_REPORT_FILES:
        src_file = report_dir / filename
        if src_file.is_file():
            _copy_file(src_file, target_dir / filename)

    for filename in UI_ASSET_FILES:
        preferred_src = ui_assets_dir / filename
        fallback_src = report_dir / filename
        if preferred_src.is_file():
            _copy_file(preferred_src, target_dir / filename)
        elif fallback_src.is_file():
            _copy_file(fallback_src, target_dir / filename)

    samples_src = report_dir / "samples"
    if samples_src.is_dir():
        shutil.copytree(samples_src, target_dir / "samples", dirs_exist_ok=True)

    if include_figures:
        figures_src = report_dir / "figures"
        if figures_src.is_dir():
            shutil.copytree(figures_src, target_dir / "figures", dirs_exist_ok=True)

    return target_dir


def main() -> None:
    """Parse CLI arguments and export the report bundle.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description=(
            "Copy generated report artifacts into spiral-web for static embedding."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("report"),
        help="Generated report directory (default: report)",
    )
    parser.add_argument(
        "--spiral-web-dir",
        type=Path,
        default=Path("../spiral-web"),
        help="Path to spiral-web repository root (default: ../spiral-web)",
    )
    parser.add_argument(
        "--target-subdir",
        type=str,
        default=DEFAULT_TARGET_SUBDIR,
        help=(f"Destination under spiral-web root (default: {DEFAULT_TARGET_SUBDIR})"),
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove destination directory before copying.",
    )
    parser.add_argument(
        "--include-figures",
        action="store_true",
        help="Also copy report/figures if present.",
    )
    args = parser.parse_args()

    exported_to = export_report_bundle(
        report_dir=args.report_dir,
        spiral_web_dir=args.spiral_web_dir,
        target_subdir=args.target_subdir,
        clean_target=not args.keep_existing,
        include_figures=args.include_figures,
    )

    print(f"Export complete: {exported_to}")


if __name__ == "__main__":
    main()
