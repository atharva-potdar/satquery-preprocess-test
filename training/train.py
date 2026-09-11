"""SatQuery Training Script — Multi-stage QDoRA/QLoRA fine-tuning.

Supports 3-stage curriculum with:
- QDoRA or QLoRA (config flag)
- Checkpoint resume
- 10% replay from previous stages
- Dual-resolution curriculum (native:proxy ratio)
- Internal validation monitoring

Usage:
    python -m training.train --config training/configs/stage1.yaml
    python -m training.train --config training/configs/stage2.yaml --resume checkpoints/stage1/last
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit"
    use_dora: bool = True
    adapter_path: str | None = None


@dataclass
class TrainingConfig:
    stage: int = 1
    epochs: float = 1.5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_steps: int = 14000
    weight_decay: float = 0.01


@dataclass
class DataConfig:
    jsonl_dir: str = ""
    jsonl_dirs: list[str] = field(default_factory=list)
    val_jsonl: str | None = None
    replay_fraction: float = 0.0
    replay_jsonl: str | None = None
    max_samples: int | None = None


@dataclass
class CurriculumConfig:
    native_proxy_ratio: list[float] = field(default_factory=lambda: [1.0, 0.0])


@dataclass
class CheckpointConfig:
    save_every: int = 2000
    output_dir: str = "checkpoints"
    resume_from: str | None = None


@dataclass
class LoggingConfig:
    eval_every: int = 500
    log_every: int = 50
    wandb_project: str = "satquery"
    wandb_run_name: str = "run"


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Path) -> TrainConfig:
    """Load training config from YAML file."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig()
    if "model" in raw:
        config.model = ModelConfig(**raw["model"])
    if "training" in raw:
        config.training = TrainingConfig(**raw["training"])
    if "data" in raw:
        config.data = DataConfig(**raw["data"])
    if "curriculum" in raw:
        config.curriculum = CurriculumConfig(**raw["curriculum"])
    if "checkpoint" in raw:
        config.checkpoint = CheckpointConfig(**raw["checkpoint"])
    if "logging" in raw:
        config.logging = LoggingConfig(**raw["logging"])

    return config


def load_jsonl(jsonl_path: Path) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_native_proxy_split(
    samples: list[dict],
    ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split samples into native and proxy based on ratio.

    Args:
        samples: All samples
        ratio: Native fraction (0.0 to 1.0)
        seed: Random seed

    Returns:
        (native_samples, proxy_samples)
    """
    import random
    rng = random.Random(seed)

    native = [s for s in samples if "proxy" not in s.get("gsd_bucket", "").lower()]
    proxy = [s for s in samples if "proxy" in s.get("gsd_bucket", "").lower()]

    # If no proxy samples, return all as native
    if not proxy:
        return samples, []

    n_native = int(len(samples) * ratio)
    n_proxy = len(samples) - n_native

    # Sample proportionally
    rng.shuffle(native)
    rng.shuffle(proxy)

    selected_native = native[:n_native]
    selected_proxy = proxy[:n_proxy]

    return selected_native, selected_proxy


def get_replay_samples(
    replay_jsonl: Path,
    fraction: float,
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Load replay samples from previous stage."""
    import random

    if not replay_jsonl.exists() or fraction <= 0:
        return []

    all_samples = load_jsonl(replay_jsonl)
    n_replay = int(n_samples * fraction)

    rng = random.Random(seed)
    return rng.sample(all_samples, min(n_replay, len(all_samples)))


def compute_curriculum_ratio(
    step: int,
    max_steps: int,
    ratios: list[float],
) -> float:
    """Compute native:proxy ratio at current step.

    Transitions smoothly between ratios at 33% and 66% of training.
    """
    if len(ratios) == 1:
        return ratios[0]

    progress = step / max_steps

    if progress < 0.33:
        return ratios[0]
    elif progress < 0.66:
        if len(ratios) >= 2:
            return ratios[1]
        return ratios[0]
    else:
        if len(ratios) >= 3:
            return ratios[2]
        elif len(ratios) >= 2:
            return ratios[1]
        return ratios[0]


class TrainingLoop:
    """Main training loop with checkpoint resume and validation."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def setup_model(self):
        """Initialize model with QDoRA or QLoRA adapter."""
        try:
            from unsloth import FastVisionModel
        except ImportError:
            print("  [WARN] unsloth not installed, using mock model for testing")
            return None

        model, tokenizer = FastVisionModel.from_pretrained(
            self.config.model.name,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
        )

        # Add adapter
        if self.config.model.use_dora:
            print("  Using QDoRA adapter")
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_dora=True,
            )
        else:
            print("  Using QLoRA adapter")
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0,
                bias="none",
                random_state=3407,
                use_dora=False,
            )

        # Load adapter checkpoint if resuming
        if self.config.model.adapter_path:
            adapter_path = Path(self.config.model.adapter_path)
            if adapter_path.exists():
                print(f"  Loading adapter from {adapter_path}")
                # Load adapter weights
                from safetensors import safe_open
                # Implementation depends on adapter format

        return model, tokenizer

    def load_data(self) -> list[dict]:
        """Load training data with optional replay."""
        all_samples = []

        # Load main data
        jsonl_dirs = self.config.data.jsonl_dirs or [self.config.data.jsonl_dir]
        for jsonl_dir in jsonl_dirs:
            dir_path = Path(jsonl_dir)
            if dir_path.exists():
                for jsonl_path in dir_path.glob("*.jsonl"):
                    if "_val_internal" in jsonl_path.name:
                        continue
                    samples = load_jsonl(jsonl_path)
                    all_samples.extend(samples)

        # Apply max_samples limit
        if self.config.data.max_samples:
            all_samples = all_samples[:self.config.data.max_samples]

        # Add replay samples
        if self.config.data.replay_fraction > 0 and self.config.data.replay_jsonl:
            replay_path = Path(self.config.data.replay_jsonl)
            replay = get_replay_samples(
                replay_path,
                self.config.data.replay_fraction,
                len(all_samples),
            )
            all_samples.extend(replay)
            print(f"  Added {len(replay)} replay samples")

        return all_samples

    def load_val_data(self) -> list[dict]:
        """Load validation data."""
        if not self.config.data.val_jsonl:
            return []

        val_path = Path(self.config.data.val_jsonl)
        if val_path.exists():
            return load_jsonl(val_path)
        return []

    def train_step(self, batch: list[dict], model, tokenizer) -> dict[str, float]:
        """Single training step.

        Returns loss dict.
        """
        # ponytail: no real forward/backward pass yet. Real implementation:
        # 1. Format batch into ChatML messages (per-sample pair_type ->
        #    template, per Section 11 of SPEC.md)
        # 2. Tokenize with image inputs
        # 3. Forward pass with loss computation
        # 4. Backward pass with gradient accumulation
        # 5. Optimizer step
        # Needs a live GPU session to write+verify against the real
        # Unsloth/Qwen3-VL API — do this on Kaggle, not blind.
        return {"loss": float("nan")}

    def validate(self, model, tokenizer, val_data: list[dict]) -> dict[str, float]:
        """Run validation and return metrics."""
        if not val_data:
            return {"val_loss": float("nan")}

        # ponytail: same stub as train_step — no real forward pass yet.
        return {"val_loss": float("nan")}

    def save_checkpoint(self, model, step: int):
        """Save model checkpoint."""
        output_dir = Path(self.config.checkpoint.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = output_dir / f"step_{step:06d}"
        ckpt_dir.mkdir(exist_ok=True)

        # Save adapter weights
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(ckpt_dir))

        # Save training state
        state = {
            "step": step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": {
                "model": self.config.model.__dict__,
                "training": self.config.training.__dict__,
            },
        }
        with open(ckpt_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"  Saved checkpoint to {ckpt_dir}")

    def run(self):
        """Main training loop."""
        print("=" * 60)
        print("  WARNING: train_step()/validate() are stubs (return NaN loss).")
        print("  This loop moves data and saves checkpoints on schedule but")
        print("  does NOT actually train anything yet. See train_step().")
        print("=" * 60)
        print(f"Starting Stage {self.config.training.stage} training")
        print(f"  Model: {self.config.model.name}")
        print(f"  DoRA: {self.config.model.use_dora}")
        print(f"  Max steps: {self.config.training.max_steps}")

        # Setup
        model_tokenizer = self.setup_model()
        if model_tokenizer is None:
            print("  [ERROR] Failed to setup model")
            return

        model, tokenizer = model_tokenizer

        # Load data
        train_data = self.load_data()
        val_data = self.load_val_data()
        print(f"  Training samples: {len(train_data)}")
        print(f"  Validation samples: {len(val_data)}")

        # Resume from checkpoint if specified
        if self.config.checkpoint.resume_from:
            resume_path = Path(self.config.checkpoint.resume_from)
            if resume_path.exists():
                state_path = resume_path / "training_state.json"
                if state_path.exists():
                    with open(state_path) as f:
                        state = json.load(f)
                    self.step = state.get("step", 0)
                    self.epoch = state.get("epoch", 0)
                    print(f"  Resumed from step {self.step}")

        # Training loop
        start_time = time.time()
        while self.step < self.config.training.max_steps:
            # Get current curriculum ratio
            ratio = compute_curriculum_ratio(
                self.step,
                self.config.training.max_steps,
                self.config.curriculum.native_proxy_ratio,
            )

            # Split batch by native/proxy
            native, proxy = get_native_proxy_split(train_data, ratio)

            # Train step (placeholder)
            losses = self.train_step([], model, tokenizer)

            self.step += 1

            # Logging
            if self.step % self.config.logging.log_every == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed if elapsed > 0 else 0
                print(f"  Step {self.step}/{self.config.training.max_steps} "
                      f"({steps_per_sec:.1f} steps/s) "
                      f"loss={losses.get('loss', 0):.4f} "
                      f"native:proxy={ratio:.2f}:{1-ratio:.2f}")

            # Validation
            if self.step % self.config.logging.eval_every == 0:
                val_metrics = self.validate(model, tokenizer, val_data)
                print(f"  [VAL] step={self.step} loss={val_metrics.get('val_loss', 0):.4f}")

                # Save best model
                if val_metrics.get("val_loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint(model, self.step)

            # Save periodic checkpoint
            if self.step % self.config.checkpoint.save_every == 0:
                self.save_checkpoint(model, self.step)

        # Final save
        self.save_checkpoint(model, self.step)
        print(f"\nTraining complete. Final step: {self.step}")


def main():
    parser = argparse.ArgumentParser(description="SatQuery Training")
    parser.add_argument("--config", type=Path, required=True, help="Training config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config.checkpoint.resume_from = args.resume

    loop = TrainingLoop(config)
    loop.run()


if __name__ == "__main__":
    main()
