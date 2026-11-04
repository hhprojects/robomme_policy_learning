"""Write the files of a finished attempt repeatedly, with no simulator, model or GPU use.

Used to find what brings NSCC nodes down during eval.py. It repeats eval.py's saving in
the same order, on saved frames from a real attempt. If this test does it, saving alone
is enough. The last line of the log shows which stage it stopped in.
"""
import dataclasses
import time
from pathlib import Path

import numpy as np

from utils import EpisodeRunLog, RolloutRecorder, write_json


@dataclasses.dataclass
class Args:
    frames_file: str = (
        "runs/test_debug/perceptual-framesamp-modul/ckpt79999/seed7/frames/PickXtimes_ep0.npz"
    )
    save_dir: str = "runs/stress_save"
    file_sets: int = 5  # reuse this many file names, so the test can't fill the disk
    minutes: float = 50  # stop after this long


def log(cycle: int, stage: str):
    print(f"{time.strftime('%H:%M:%S')} cycle {cycle} {stage}")


def stress(args: Args):
    save_dir = Path(args.save_dir)
    data = np.load(args.frames_file)
    images, wrist_images, states = data["images"], data["wrist_images"], data["states"]
    video_frames = list(np.concatenate([images, wrist_images], axis=2))

    deadline = time.monotonic() + args.minutes * 60
    cycle = 0

    while time.monotonic() < deadline:
        episode_id = cycle % args.file_sets

        log(cycle, "video")
        recorder = RolloutRecorder(save_dir / "videos", "stress test")
        recorder.total_images = video_frames
        recorder.save_video(f"ep{episode_id}.mp4")

        log(cycle, "run log and frames")
        run_log = EpisodeRunLog("PickXtimes", episode_id, save_frames=True)
        run_log._frames = {
            "images": list(images),
            "wrist_images": list(wrist_images),
            "states": list(states),
        }
        run_log.save(save_dir)

        log(cycle, "progress")
        write_json(save_dir / "progress.json", {"cycle": cycle}, indent=2)
        cycle += 1

    print(f"finished {cycle} cycles with no failure")


if __name__ == "__main__":
    import tyro
    tyro.cli(stress)
