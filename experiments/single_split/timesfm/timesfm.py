from typing import Dict, Any, List

from src.base.torch_single_split_experiment import (
    TorchSingleSplitExperiment,
    run_single_split_experiment,
)
from src.models.time_series.formers.timesfm_model import TimesFM
from src.engines.timesfm_engine import TimesFM_Engine
from src.utils.args import get_timesfm_config
from src.utils.setup import setup_seed, setup_dataloader


class TimesFMExperiment(TorchSingleSplitExperiment):
    """TimesFM zero-shot inference experiment."""

    def get_config_parser(self):
        return get_timesfm_config()

    def get_model_name(self):
        return "timesfm"

    def get_engine_class(self):
        return TimesFM_Engine

    # def get_log_dir_suffix(self, args):
    #     return "timesfm_zeroshot"

    def get_log_dir_params(self, args):
        return f"seq_len_{args.seq_len}_pred_len_{args.pred_len}_bs_{args.batch_size}"

    def get_metrics(self) -> List[str]:
        return ["mse", "mae", "mape", "rmse"]

    def create_model(self, args, dataloader):
        self._input_channels = self.dataloader_instance.get_input_channels(
            getattr(args, "input_channels", None)
        )

        return TimesFM(
            input_channels=self._input_channels,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            # model_name=getattr(args, "timesfm_model_name", "google/timesfm-1.0-200m-pytorch"),
            model_name=getattr(args, "timesfm_model_name", "google/timesfm-2.5-200m-pytorch"),
            backend=getattr(args, "timesfm_backend", "gpu"),
            per_core_batch_size=getattr(args, "timesfm_per_core_batch_size", args.batch_size),
        )

    def run(self):
        """Run zero-shot evaluation."""
        args, log_dir, logger = self.get_config()

        setup_seed(args.seed)

        self.dataloader_instance, self.dataloader = setup_dataloader(
            dataloader_class=self.get_dataloader_class(),
            dataloader_kwargs=self.get_dataloader_kwargs(args),
        )
        self.scaler = self.dataloader_instance.get_scaler()

        self.model = self.create_model(args, self.dataloader)
        self.model.print_experiment_summary(args, logger)

        # no optimizer needed
        loss_fn = self.get_loss_function()
        self.engine = self.create_engine(
            args=args,
            model=self.model,
            dataloader=self.dataloader,
            scaler=self.scaler,
            optimizer=None,
            scheduler=None,
            loss_fn=loss_fn,
            log_dir=log_dir,
            logger=logger,
        )

        result = self.engine.train()
        logger.info(f"Result: {result}")

        return result


if __name__ == "__main__":
    run_single_split_experiment(TimesFMExperiment)
