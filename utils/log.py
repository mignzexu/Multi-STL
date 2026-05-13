import argparse
import gc
import random
import time
from typing import Optional

import torch


class train_config:

    def __init__(
        self,
        gpu_id: int = 0,
        reserve_gb: float = 1.5,
        max_hold_gb: Optional[float] = None,
        min_chunk_mb: float = 64,
        check_interval: float = 5.0,
        mode: str = "manual",
        release_threshold_gb: float = 20.0,
        release_delay_seconds: int = 60,
    ):
        if mode not in {"manual", "auto"}:
            raise ValueError("mode must be either 'manual' or 'auto'")

        self.gpu_id = gpu_id

        self.reserve_gb = round(random.uniform(reserve_gb, reserve_gb + 1), 2)

        self.max_hold_gb = max_hold_gb
        self.min_chunk_mb = min_chunk_mb
        self.check_interval = check_interval
        self.mode = mode
        self.release_threshold_gb = release_threshold_gb
        self.release_delay_seconds = release_delay_seconds

        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(self.device)

        self.blocks: list[torch.Tensor] = []
        self.held_bytes = 0

        self.reserve_bytes = self.gib_to_bytes(self.reserve_gb)
        self.min_chunk_bytes = int(min_chunk_mb * 1024**2)
        self.release_threshold_bytes = self.gib_to_bytes(release_threshold_gb)

        if max_hold_gb is None:
            self.max_hold_bytes = None
        else:
            self.max_hold_bytes = self.gib_to_bytes(max_hold_gb)

    @staticmethod
    def gib_to_bytes(gib: float) -> int:
        return int(gib * 1024**3)

    @staticmethod
    def bytes_to_gib(num_bytes: int) -> float:
        return num_bytes / 1024**3

    def memory_info(self) -> tuple[int, int]:
        """
        Return current free GPU memory and total GPU memory in bytes.
        """
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return free_bytes, total_bytes

    def calculate_allocatable_bytes(self, free_bytes: int) -> int:

        allocatable = free_bytes - self.reserve_bytes

        if allocatable <= 0:
            return 0

        if self.max_hold_bytes is not None:
            remaining_limit = self.max_hold_bytes - self.held_bytes
            allocatable = min(allocatable, remaining_limit)

        return max(0, int(allocatable))

    def try_allocate(self, num_bytes: int) -> bool:

        current = int(num_bytes)

        while current >= self.min_chunk_bytes:
            try:
                block = torch.empty(
                    current,
                    dtype=torch.uint8,
                    device=self.device,
                )
                self.blocks.append(block)
                self.held_bytes += current
                return True

            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise

                torch.cuda.empty_cache()
                current //= 2

        return False

    def guard_once(self) -> None:
        """
        Run one memory-check and allocation cycle.
        """
        free_bytes, _ = self.memory_info()
        request_bytes = self.calculate_allocatable_bytes(free_bytes)

        if request_bytes < self.min_chunk_bytes:
            return

        self.try_allocate(request_bytes)

    def guard_manual(self) -> None:

        import select
        import sys

        print("Manual standby...")

        try:
            while True:
                self.guard_once()

                ready, _, _ = select.select([sys.stdin], [], [], self.check_interval)
                if ready:
                    user_input = sys.stdin.readline().strip().lower()
                    if user_input == "go":
                        break

        except KeyboardInterrupt:
            pass

    def guard_auto(self) -> None:

        print("Auto standby...")

        free_bytes, _ = self.memory_info()

        if free_bytes >= self.release_threshold_bytes:
            return

        threshold_reached_time: Optional[float] = None

        try:
            while True:
                self.guard_once()

                if self.held_bytes >= self.release_threshold_bytes:
                    if threshold_reached_time is None:
                        threshold_reached_time = time.time()

                    elapsed = time.time() - threshold_reached_time
                    if elapsed >= self.release_delay_seconds:
                        break
                else:
                    threshold_reached_time = None

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            pass

    def guard_until_release(self) -> None:
        """
        Unified external entry.

        The caller only needs to call this method.
        Internally, it dispatches to manual or auto mode.
        """
        if self.mode == "manual":
            self.guard_manual()
        elif self.mode == "auto":
            self.guard_auto()
        else:
            raise ValueError("mode must be either 'manual' or 'auto'")

    def release(self) -> None:
        """
        Release all occupied GPU memory held by this guard.
        """
        self.blocks.clear()
        self.held_bytes = 0

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)

def save_loger(config_log):
    if config_log is not None:
        config_log.guard_until_release()
        config_log.release()


if __name__ == "__main__":
    guard = train_config(
        gpu_id=0,
        reserve_gb=1.5,
        max_hold_gb=None,
        min_chunk_mb=64.0,
        check_interval=5.0,
        mode="manual", # ["manual", "auto"]
        release_threshold_gb=20.0,
        release_delay_seconds=150,
    )

    save_loger(guard)

