from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torchmetrics.functional.regression import mean_absolute_error

from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

def r2score(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    R² score 계산 함수.
    R² = 1 - (SS_res / SS_tot)
    """
    target = target.float()
    preds = preds.float()
    target_mean = target.mean()

    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - preds) ** 2).sum()

    return 1 - ss_res / ss_tot


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """Attaches a MLP for fine-tuning using the standard self-supervised protocol.

    Example::

        # your datamodule must have 2 attributes
        dm = DataModule()
        dm.num_classes = ... # the num of classes in the datamodule
        dm.name = ... # name of the datamodule (e.g. ImageNet, STL10, CIFAR10)

        # your model must have 1 attribute
        model = Model()
        model.z_dim = ... # the representation dim

        online_eval = SSLOnlineEvaluator(
            z_dim=model.z_dim
        )
    """

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        swav: bool = False,
        multimodal: bool = False,
        strategy: str = None,
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = None
        self.dataset: Optional[str] = None
        self.num_classes: Optional[int] = num_classes
        self.swav = swav
        self.multimodal = multimodal
        self.strategy = strategy

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        # switch fo PL compatibility reasons
        # accel = (
        #     trainer.accelerator_connector
        #     if hasattr(trainer, "accelerator_connector")
        #     else trainer._accelerator_connector
        # )
        # if accel.is_distributed:
        #     if accel.use_ddp:
        #         from torch.nn.parallel import DistributedDataParallel as DDP

        #         self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
        #     elif accel.use_dp:
        #         from torch.nn.parallel import DataParallel as DP

        #         self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
        #     else:
        #         rank_zero_warn(
        #             "Does not support this type of distributed accelerator. The online evaluator will not sync."
        #         )

        accel = (
            trainer.accelerator_connector
            if hasattr(trainer, "accelerator_connector")
            else trainer._accelerator_connector
        )
        # print(trainer.accelerator)
        if accel.is_distributed:
            if accel._strategy_flag in ["ddp", "ddp2", "ddp_spawn"]:
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.online_evaluator = DDP(self.online_evaluator, device_ids=[pl_module.device])
            elif trainer._strategy_flag == "dp":
                from torch.nn.parallel import DataParallel as DP

                self.online_evaluator = DP(self.online_evaluator, device_ids=[pl_module.device])
            else:
                rank_zero_warn(
                    "Does not support this type of distributed accelerator. The online evaluator will not sync."
                )

    #    if trainer.is_global_zero:
    #         self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

    #     if trainer.use_ddp2:
    #         from torch.nn.parallel import DistributedDataParallel as DDP

    #         self.online_evaluator = DDP(self.online_evaluator, device_ids=[trainer.local_rank], broadcast_buffers=False)
    #     elif trainer.use_dp:
    #         from torch.nn.parallel import DataParallel as DP

    #         self.online_evaluator = DP(self.online_evaluator, device_ids=[trainer.root_device])
    #     else:
    #         rank_zero_warn(
    #             "Does not support this type of distributed accelerator. The online evaluator will not sync."
    #         )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(self._recovered_callback_state["state_dict"])
            self.optimizer.load_state_dict(self._recovered_callback_state["optimizer_state"])

    def to_device(self, batch: Sequence, device: Union[str, torch.device]) -> Tuple[Tensor, Tensor]:

        if self.swav:
            x, y = batch
            x = x[0]

        elif self.multimodal and self.strategy == 'comparison':
            x_i, _, y, x_orig = batch
            x = x_orig

        elif self.multimodal and self.strategy == 'tip':
            x_i, _, y, x_orig, x_t_orig = batch 
            x = x_orig
            x_t = x_t_orig

        else:
            _, x, y = batch
        
        if self.strategy == 'comparison':
            # last input is for online eval
            x = x.to(device)
            y = y.to(device)

            return x, y, None

        elif self.strategy == 'tip':
            x = x.to(device)
            y = y.to(device)
            x_t = x_t.to(device)

            return x, y, x_t

        else:
            Exception('Strategy must be comparison or tip')

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y, x_t = self.to_device(batch, pl_module.device)
                representations = pl_module(x, tabular=x_t) if x_t is not None else pl_module(x)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        if mlp_logits.dim() > 1 and mlp_logits.size(-1) == 1:
            mlp_logits = mlp_logits.squeeze(-1)

        mlp_loss = F.mse_loss(mlp_logits, y)

        r2 = r2score(mlp_logits, y)
        mae = mean_absolute_error(mlp_logits, y)

        return r2, mae, mlp_loss

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int
    ) -> None:
        train_r2, train_mae, mlp_loss = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("regressor.train.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("regressor.train.r2", train_r2, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("regressor.train.mae", train_mae, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        val_r2, val_mae, mlp_loss = self.shared_step(pl_module, batch)
        pl_module.log("regressor.val.loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("regressor.val.r2", val_r2, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("regressor.val.mae", val_mae, on_step=False, on_epoch=True, sync_dist=True)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> dict:
        return {"state_dict": self.online_evaluator.state_dict(), "optimizer_state": self.optimizer.state_dict()}

    def on_load_checkpoint(self, trainer: Trainer, pl_module: LightningModule, callback_state: Dict[str, Any]) -> None:
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
