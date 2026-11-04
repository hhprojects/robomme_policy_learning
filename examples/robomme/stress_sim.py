"""Open, draw and close the simulator repeatedly, with no model and no saving.

Used to find what brings NSCC nodes down during eval.py. If this test does it, the
simulator alone is enough. The last line of the log shows which stage it stopped in.
"""
import dataclasses
import time
from typing import Tuple

from env_runner import EnvRunner


@dataclasses.dataclass
class Args:
    tasks: Tuple[str, ...] = ("PickXtimes", "BinFill", "StopCube")
    steps_per_cycle: int = 20  # simulator steps in each cycle, each one draws a frame
    minutes: float = 50  # stop after this long
    max_steps: int = 1300


def log(cycle: int, task_name: str, episode_id: int, stage: str):
    print(f"{time.strftime('%H:%M:%S')} cycle {cycle} {task_name} ep{episode_id} {stage}")


def stress(args: Args):
    deadline = time.monotonic() + args.minutes * 60
    cycle = 0

    while True:
        for task_name in args.tasks:
            env_runner = EnvRunner(task_name, "", max_steps=args.max_steps)

            for episode_id in range(env_runner.num_episodes):
                if time.monotonic() > deadline:
                    print(f"finished {cycle} cycles with no failure")
                    return

                log(cycle, task_name, episode_id, "open")
                env_runner.make_env(episode_id)
                log(cycle, task_name, episode_id, "reset")
                action = env_runner.get_init_obs()["states"][-1]

                log(cycle, task_name, episode_id, "draw")
                for _ in range(args.steps_per_cycle):
                    (_, _, action), stop, _ = env_runner.step(action)
                    if stop:
                        break

                log(cycle, task_name, episode_id, "close")
                env_runner.close_env()
                cycle += 1


if __name__ == "__main__":
    import tyro
    tyro.cli(stress)
