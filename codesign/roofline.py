"""
Roofline model analyzer.

Generates a roofline analysis showing which layers are compute-bound
vs memory-bound for a given hardware target, and plots the roofline
chart using matplotlib (or falls back to ASCII art).
"""

from __future__ import annotations
from typing import List, Optional
import sys

from .architecture import NNArchitecture
from .hardware import HardwareSpec
from .estimator import PerformanceEstimator, LayerResult, PerformanceResult


class RooflineAnalyzer:
    """
    Roofline model visualization and analysis.

    Usage:
        analyzer = RooflineAnalyzer()
        result = analyzer.analyze(arch, hw)
        analyzer.print_report(result)
        analyzer.plot(result)   # requires matplotlib
    """

    def __init__(self, estimator: Optional[PerformanceEstimator] = None):
        self.estimator = estimator or PerformanceEstimator()

    def analyze(self, arch: NNArchitecture, hw: HardwareSpec) -> PerformanceResult:
        return self.estimator.estimate(arch, hw)

    def print_report(self, result: PerformanceResult, verbose: bool = False) -> None:
        print(result.summary())
        print()

        ridge = result.layer_results[0].ridge_point if result.layer_results else 0.0
        print(f"  Ridge point: {ridge:.1f} FLOP/byte")
        print()

        # Classify layers
        compute_bound = [r for r in result.layer_results if r.is_compute_bound]
        memory_bound = [r for r in result.layer_results if not r.is_compute_bound]

        print(f"  Compute-bound layers ({len(compute_bound)}):")
        for r in sorted(compute_bound, key=lambda x: -x.latency_ms)[:5]:
            self._print_layer_row(r)

        print(f"\n  Memory-bound layers ({len(memory_bound)}):")
        for r in sorted(memory_bound, key=lambda x: -x.latency_ms)[:5]:
            self._print_layer_row(r)

        if verbose:
            print("\n  All layers:")
            for r in result.layer_results:
                self._print_layer_row(r)

    def _print_layer_row(self, r: LayerResult) -> None:
        bound = "COMPUTE" if r.is_compute_bound else "MEMORY "
        sram = "✓" if r.fits_in_sram else "✗"
        print(
            f"    [{bound}] {r.name:<28} "
            f"AI={r.arith_intensity:>8.1f}  "
            f"lat={r.latency_ms:>8.4f}ms  "
            f"SRAM={sram}"
        )

    def print_ascii_chart(self, result: PerformanceResult) -> None:
        """
        ASCII roofline chart: x-axis = log10(arithmetic intensity),
        y-axis = log10(attainable FLOP/s).
        """
        import math

        lr = result.layer_results
        if not lr:
            return

        ridge = lr[0].ridge_point
        peak = lr[0].attainable_flops / (lr[0].arith_intensity or 1) * ridge  # peak flops
        # Recover hw peak from first layer
        bw = lr[0].attainable_flops  # This may be inaccurate; use stored ridge instead

        WIDTH, HEIGHT = 70, 20

        # x: log10(AI) from -1 to 5
        # y: log10(FLOP/s) from 6 to 15
        x_min, x_max = -1.0, 5.0
        y_min, y_max = 6.0, 15.0

        grid = [[' '] * WIDTH for _ in range(HEIGHT)]

        def to_xy(ai, flops_s):
            if ai <= 0 or flops_s <= 0:
                return None, None
            xi = (math.log10(ai) - x_min) / (x_max - x_min)
            yi = (math.log10(flops_s) - y_min) / (y_max - y_min)
            col = int(xi * (WIDTH - 1))
            row = HEIGHT - 1 - int(yi * (HEIGHT - 1))
            if 0 <= col < WIDTH and 0 <= row < HEIGHT:
                return row, col
            return None, None

        # Draw roofline (theoretical ceiling)
        hw_peak_flops = max(r.attainable_flops for r in lr)
        hw_bw = hw_peak_flops / ridge if ridge > 0 else 1.0

        for x_frac in range(WIDTH):
            ai = 10 ** (x_min + x_frac * (x_max - x_min) / (WIDTH - 1))
            roof = min(hw_peak_flops, hw_bw * ai)
            row, col = to_xy(ai, roof)
            if row is not None:
                grid[row][col] = '─'

        # Draw ridge point vertical
        ridge_col = int((math.log10(max(ridge, 1e-1)) - x_min) / (x_max - x_min) * (WIDTH - 1))
        if 0 <= ridge_col < WIDTH:
            for row in range(HEIGHT):
                if grid[row][ridge_col] == ' ':
                    grid[row][ridge_col] = '┊'

        # Plot each layer
        symbols = {True: '●', False: '○'}
        for r in lr:
            if r.flops == 0:
                continue
            row, col = to_xy(r.arith_intensity, r.attainable_flops)
            if row is not None:
                grid[row][col] = symbols[r.is_compute_bound]

        print(f"\n  Roofline Chart — {result.hw_name}")
        print(f"  ● compute-bound  ○ memory-bound  ┊ ridge={ridge:.1f} FLOP/byte")
        print()
        print(f"  10^{y_max:.0f} |" + "─" * WIDTH)
        for i, row in enumerate(grid):
            y_val = y_max - i * (y_max - y_min) / (HEIGHT - 1)
            prefix = f"  10^{y_val:.1f} |" if i % 4 == 0 else "         |"
            print(prefix + "".join(row))
        print("         +" + "─" * WIDTH)
        print(f"           10^{x_min} {'AI (FLOP/byte)':^{WIDTH - 10}} 10^{x_max}")

    def plot(self, result: PerformanceResult, save_path: Optional[str] = None) -> None:
        """
        Matplotlib roofline plot. Falls back to ASCII if matplotlib unavailable.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("[RooflineAnalyzer] matplotlib not available — using ASCII chart")
            self.print_ascii_chart(result)
            return

        lr = result.layer_results
        if not lr:
            return

        ridge = lr[0].ridge_point
        hw_peak = max(r.attainable_flops for r in lr)
        hw_bw = hw_peak / ridge if ridge > 0 else 1.0

        ai_vals = np.logspace(-1, 5, 500)
        roof = np.minimum(hw_peak, hw_bw * ai_vals)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(ai_vals, roof, 'k-', linewidth=2, label='Roofline')
        ax.axvline(ridge, color='gray', linestyle='--', alpha=0.5, label=f'Ridge={ridge:.1f}')

        compute_pts = [(r.arith_intensity, r.attainable_flops, r.name) for r in lr if r.is_compute_bound and r.flops > 0]
        memory_pts  = [(r.arith_intensity, r.attainable_flops, r.name) for r in lr if not r.is_compute_bound and r.flops > 0]

        if compute_pts:
            ax.scatter([p[0] for p in compute_pts], [p[1] for p in compute_pts],
                       c='tab:blue', marker='o', s=60, label='Compute-bound', zorder=5)
        if memory_pts:
            ax.scatter([p[0] for p in memory_pts], [p[1] for p in memory_pts],
                       c='tab:orange', marker='s', s=60, label='Memory-bound', zorder=5)

        ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
        ax.set_ylabel('Attainable FLOP/s')
        ax.set_title(f'Roofline: {result.arch_name} on {result.hw_name}')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"[RooflineAnalyzer] Saved plot to {save_path}")
        else:
            plt.show()
        plt.close()
