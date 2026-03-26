import torch
import numpy as np
from typing import Dict
from collections import defaultdict

from src.base.torch_engine import TorchEngine


class TimesFM_Engine(TorchEngine):
    """
    TimesFM Engine for zero-shot inference only.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger.info("TimesFM running in zero-shot mode (inference only)")

    def train(self) -> Dict[str, float]:
        """Skip training, go directly to evaluation."""
        self._logger.info("=" * 60)
        self._logger.info("TimesFM Zero-Shot Evaluation")
        self._logger.info("=" * 60)

        # Evaluate on test set
        self._logger.info("\nEvaluating on test set...")

        test_metrics = self._run_test_evaluation()

        self._logger.info("=" * 60)
        self._logger.info("Zero-shot evaluation completed")
        self._logger.info("=" * 60)

        return test_metrics

    def _run_test_evaluation(self) -> Dict[str, float]:
        """Run evaluation on test set."""
        self.model.eval()

        preds, labels = [], []

        with torch.no_grad():
            for batch in self._get_dataloader("test_loader"):
                batch_dict = self._prepare_batch(batch)
                batch_dict = self._to_device(batch_dict)

                x_batch = batch_dict["x"]
                y_batch = batch_dict["y"]

                # Forward pass
                pred = self.model(x_batch, flatten_output=False)

                preds.append(pred.cpu())
                labels.append(y_batch.cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        # Compute and log metrics
        return self._compute_test_metrics(preds, labels)
