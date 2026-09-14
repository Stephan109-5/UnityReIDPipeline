from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Sequence

from src.metrics.trackeval_offline import evaluate_run_folder


def _evaluate_folder(folder: Path) -> None:
    print(f'Evaluating TrackEval results in {folder}...')
    evaluate_run_folder(folder)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Evaluate multiple saved runs with TrackEval in parallel.')
    parser.add_argument('parent_folder', type=Path, help='Folder containing run subfolders to evaluate.')
    args = parser.parse_args(argv)

    run_folders = [
        folder
        for folder in sorted(args.parent_folder.iterdir())
        if folder.is_dir() and ((folder / 'tracking_results').exists() or (folder / 'reid_results').exists())
    ]
    if not run_folders:
        print(f'No run folders found in {args.parent_folder}.')
        return 0

    processes: list[mp.Process] = []
    for folder in run_folders:
        process = mp.Process(target=_evaluate_folder, args=(folder,), name=f'trackeval-{folder.name}')
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())