"""TimeMixerPP Engine for training and evaluation."""

import torch
from src.base.torch_engine import TorchEngine


class TimeMixerPP_Engine(TorchEngine):
    """Engine for TimeMixerPP model training and evaluation."""

    def __init__(
        self,
        model,
        dataloader,
        scaler,
        optimizer,
        scheduler,
        loss_fn,
        log_dir,
        logger,
        clip_grad_value=0,
        max_epochs=100,
        patience=10,
        device="cuda",
        **kwargs,
    ):
        super().__init__(
            model=model,
            dataloader=dataloader,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            log_dir=log_dir,
            logger=logger,
            clip_grad_value=clip_grad_value,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            **kwargs,
        )

    def _train_batch(self, batch):
        """Process a single training batch."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(x)

        # Compute loss
        loss = self.loss_fn(output, y)

        # Backward pass
        loss.backward()

        if self.clip_grad_value > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_value)

        self.optimizer.step()

        return loss.item()

    def _eval_batch(self, batch):
        """Process a single evaluation batch."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            output = self.model(x)
            loss = self.loss_fn(output, y)

        return output, y, loss.item()
