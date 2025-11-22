#!/usr/bin/env python3
"""
Training Monitor for Finnish TTS
Real-time monitoring of Fish Speech training progress

Usage:
    python monitor_training.py [--watch]
    
Options:
    --watch: Continuously monitor training (updates every 10 seconds)
"""

import argparse
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
import subprocess


class TrainingMonitor:
    def __init__(self, results_dir="results/FinnishSpeaker_2000_finetune"):
        self.results_dir = Path(results_dir)
        self.checkpoint_dir = self.results_dir / "checkpoints"
        self.log_file = self.results_dir / "train.log"
        
    def get_checkpoints(self):
        """Get list of checkpoint files"""
        if not self.checkpoint_dir.exists():
            return []
        
        checkpoints = list(self.checkpoint_dir.glob("step_*.ckpt"))
        return sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    
    def get_latest_logs(self, lines=30):
        """Get latest log lines"""
        if not self.log_file.exists():
            return []
        
        with open(self.log_file, 'r') as f:
            all_lines = f.readlines()
            return all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    def get_gpu_stats(self):
        """Get current GPU utilization"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                stats = result.stdout.strip().split(', ')
                return {
                    'gpu_util': float(stats[0]),
                    'mem_util': float(stats[1]),
                    'mem_used': float(stats[2]),
                    'mem_total': float(stats[3]),
                    'temp': float(stats[4])
                }
        except:
            pass
        return None
    
    def parse_loss_from_logs(self, logs):
        """Extract loss values from logs"""
        import re
        losses = []
        for line in logs:
            # Adjust regex based on your actual log format
            match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
            if match:
                try:
                    losses.append(float(match.group(1)))
                except:
                    pass
        return losses
    
    def display_status(self, clear_screen=True):
        """Display current training status"""
        if clear_screen:
            print("\033[2J\033[H")  # Clear screen
        
        print("=" * 70)
        print(f"{'🎯 FINNISH TTS TRAINING MONITOR':^70}")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Checkpoints
        checkpoints = self.get_checkpoints()
        if checkpoints:
            steps = [int(ckpt.stem.split('_')[-1]) for ckpt in checkpoints]
            latest_step = max(steps)
            latest_ckpt = checkpoints[-1]
            
            print("📊 TRAINING PROGRESS")
            print("-" * 70)
            print(f"Total Checkpoints:  {len(checkpoints)}")
            print(f"Current Step:       {latest_step:,}")
            print(f"Latest Checkpoint:  {latest_ckpt.name}")
            
            # Calculate progress
            target_steps = 2000
            progress = (latest_step / target_steps) * 100
            bar_length = 40
            filled = int(bar_length * latest_step / target_steps)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"\nProgress: [{bar}] {progress:.1f}%")
            print(f"Remaining: {target_steps - latest_step:,} steps")
            
            # Estimate time
            if len(checkpoints) > 1:
                first_ckpt = checkpoints[0]
                time_elapsed = latest_ckpt.stat().st_mtime - first_ckpt.stat().st_mtime
                steps_done = latest_step - int(first_ckpt.stem.split('_')[-1])
                if steps_done > 0:
                    time_per_step = time_elapsed / steps_done
                    remaining_time = time_per_step * (target_steps - latest_step)
                    print(f"Est. Time Remaining: {timedelta(seconds=int(remaining_time))}")
        else:
            print("📊 TRAINING PROGRESS")
            print("-" * 70)
            print("No checkpoints found yet. Training may not have started.")
        
        print()
        
        # Loss metrics
        logs = self.get_latest_logs(50)
        losses = self.parse_loss_from_logs(logs)
        if losses:
            print("📈 LOSS METRICS (Recent)")
            print("-" * 70)
            print(f"Current:  {losses[-1]:.6f}")
            print(f"Min:      {min(losses):.6f}")
            print(f"Max:      {max(losses):.6f}")
            print(f"Average:  {sum(losses)/len(losses):.6f}")
            
            # Simple trend
            if len(losses) >= 10:
                recent_avg = sum(losses[-10:]) / 10
                earlier_avg = sum(losses[-20:-10]) / 10 if len(losses) >= 20 else recent_avg
                trend = "↓ Improving" if recent_avg < earlier_avg else "↑ Increasing"
                print(f"Trend:    {trend}")
        
        print()
        
        # GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print("💻 GPU STATUS")
            print("-" * 70)
            print(f"GPU Utilization:    {gpu_stats['gpu_util']:.0f}%")
            print(f"Memory Utilization: {gpu_stats['mem_util']:.0f}%")
            print(f"Memory Used:        {gpu_stats['mem_used']:.0f} / {gpu_stats['mem_total']:.0f} MB")
            print(f"Temperature:        {gpu_stats['temp']:.0f}°C")
        
        print()
        
        # Recent log lines
        print("📝 RECENT LOGS")
        print("-" * 70)
        if logs:
            for line in logs[-10:]:
                line = line.strip()
                if line:
                    print(f"  {line[:68]}")
        else:
            print("  No logs available yet.")
        
        print()
        print("=" * 70)
    
    def watch(self, interval=10):
        """Continuously monitor training"""
        print("Starting training monitor...")
        print(f"Refreshing every {interval} seconds. Press Ctrl+C to stop.\n")
        
        try:
            while True:
                self.display_status(clear_screen=True)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Monitor Finnish TTS training")
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuously monitor training (updates every 10 seconds)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Update interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/FinnishSpeaker_2000_finetune',
        help='Training results directory'
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(results_dir=args.results_dir)
    
    if args.watch:
        monitor.watch(interval=args.interval)
    else:
        monitor.display_status(clear_screen=False)


if __name__ == "__main__":
    main()
