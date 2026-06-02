# MeanTeacher wrapper module: provides a single object that contains student and teacher models,
# performs student backward + optimizer.step, updates teacher via EMA, computes losses, and applies augmentations.
# This is a self-contained implementation (pure PyTorch, no torchvision).

import copy
from typing import Callable, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

try:
    from bm import BM
except:
    from .bm import BM

# --------------------- Utilities ---------------------

def default_consistency_criterion(student_out: torch.Tensor, teacher_out: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student_out, teacher_out)

# def default_unlabeled_aug(img: torch.Tensor, brightness=0.2, contrast=0.2, noise_std=0.03) -> torch.Tensor:
#     """
#     Simple in-place style augmentation for unlabeled images. Expects img in [0,1], shape CxHxW.
#     Does brightness, contrast, and additive gaussian noise. Deterministic per call.
#     """
#     # brightness
#     factor = 1.0 + random.uniform(-brightness, brightness)
#     img = img * factor
#     # contrast (scale about mean)
#     mean = img.mean(dim=[1,2], keepdim=True)
#     c_factor = 1.0 + random.uniform(-contrast, contrast)
#     img = (img - mean) * c_factor + mean
#     # noise
#     if noise_std > 0:
#         std = random.uniform(0.0, noise_std)
#         img = img + torch.randn_like(img) * std
#     return img.clamp(0, 1.0)

# def default_labeled_aug(img: torch.Tensor, noise_std=0.01) -> torch.Tensor:
#     """
#     Mild augmentations allowed for labeled images that preserve spatial alignment of density maps.
#     """
#     if noise_std > 0:
#         std = random.uniform(0.0, noise_std)
#         img = img + torch.randn_like(img) * std
#     return img.clamp(0, 1.0)

# --------------------- MeanTeacherWrapper ---------------------

class MeanTeacherWrapper(nn.Module):
    """
    Wraps a student model and a teacher model (EMA of student). Provides:
      - forward_supervised: compute supervised loss on labeled batch
      - forward_unlabeled: compute consistency loss on unlabeled batch (student vs teacher with different augs)
      - training_step: perform combined step (compute losses, backward on student, optimizer.step(), EMA update)
      - update_ema: manually update teacher parameters
      - predict_student / predict_teacher: inference helpers
    """
    def __init__(
        self,
        student: nn.Module,
        ema_decay: float = 0.999,
        consistency_criterion: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        unlabeled_augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = nn.Identity(),
        labeled_augmentation: Optional[Callable[[torch.Tensor], torch.Tensor]] = nn.Identity(),
        consistency_rampup_epochs: int = 5,
        max_consistency_weight: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        # student is the actual trainable model
        self.student = student
        # teacher is EMA copy of student parameters (not trainable)
        self.teacher = copy.deepcopy(student)
        self._set_requires_grad(self.teacher, False)
        self.ema_decay = ema_decay
        self.consistency_criterion = consistency_criterion or default_consistency_criterion
        self.unlabeled_augmentation = unlabeled_augmentation#  or default_unlabeled_aug
        self.labeled_augmentation = labeled_augmentation#  or default_labeled_aug
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.max_consistency_weight = max_consistency_weight
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.to(self.device)
        # internal counters
        self._current_epoch = 0
        self._global_step = 0

    def _set_requires_grad(self, model: nn.Module, req: bool):
        for p in model.parameters():
            p.requires_grad = req

    def update_ema(self):
        """
        Update teacher parameters with EMA from student.
        """
        sd = self.student.state_dict()
        td = self.teacher.state_dict()
        for k in td.keys():
            if td[k].dtype in (torch.float16, torch.float32, torch.float64):
                td[k].mul_(self.ema_decay).add_(sd[k], alpha=1.0 - self.ema_decay)
            else:
                td[k].copy_(sd[k])
        self.teacher.load_state_dict(td)

    def _consistency_weight(self) -> float:
        if self.consistency_rampup_epochs <= 0:
            return self.max_consistency_weight
        p = float(min(self._current_epoch, self.consistency_rampup_epochs)) / float(self.consistency_rampup_epochs)
        return float(self.max_consistency_weight * p)

    def forward_supervised(self, 
                           imgs: torch.Tensor, 
                           targets: torch.Tensor, 
                           bayesian_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                           st_sizes,
                           points):
        """
        Compute supervised predictions and loss on labeled batch.
        Returns (preds, sup_loss).
        imgs: BxCxHxW, targets: ground-truth density maps (same spatial dims as preds expected by model)
        bayesian_loss: callable(preds, targets) -> scalar tensor
        """
        imgs[0] = imgs[0].to(self.device)
        imgs[1] = imgs[1].to(self.device)
        # mild labeled augmentation that preserves spatial correspondence

        if targets is None and bayesian_loss is None:
            return self.student(imgs)
        else:
            targets = [t.to(self.device) for t in targets]
            imgs_aug = torch.stack([self.labeled_augmentation(x) for x in imgs])
            preds = self.student(imgs_aug)
            sup_loss = bayesian_loss(points, st_sizes, targets, preds)
            return preds, sup_loss

    def forward_unlabeled(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute student and teacher predictions for unlabeled images with different augmentations.
        Returns (student_preds, teacher_preds).
        """
        imgs[0] = imgs[0].to(self.device)
        imgs[1] = imgs[1].to(self.device)
        # create two independent augmented views
        student_views = torch.stack([self.unlabeled_augmentation(x) for x in imgs]).to(self.device)
        teacher_views = torch.stack([self.unlabeled_augmentation(x) for x in imgs]).to(self.device)

        student_preds = self.student(student_views)
        # teacher in eval/no-grad mode
        with torch.no_grad():
            teacher_preds = self.teacher(teacher_views)
        return student_preds, teacher_preds

    def training_step(
        self,
        labeled_batch: Tuple[torch.Tensor, torch.Tensor],
        unlabeled_batch: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        bayesian_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        supervised_weight: float = 1.0,
        clip_grad: Optional[float] = None,
        isLabel = True
    ) -> dict:
        """
        Perform a single training step:
          - compute supervised loss on labeled_batch
          - compute consistency loss on unlabeled_batch
          - backprop supervised + weighted consistency on student
          - optimizer.step() to update student
          - update teacher by EMA
        Returns a dict with losses and metrics.
        """
        self.student.train()
        self.teacher.eval()

        imgs_l, targets, st_sizes, points = labeled_batch
        imgs_u = unlabeled_batch

        # supervised forward
        if isLabel:
            preds_l, sup_loss = self.forward_supervised(imgs_l, targets, bayesian_loss, st_sizes, points)
        else:
            preds_l = None
            sup_loss = torch.tensor(0.0, device=self.device)

        # unlabeled forward (student vs teacher)
        student_u_preds, teacher_u_preds = self.forward_unlabeled(imgs_u)
        cons_loss = self.consistency_criterion(student_u_preds, teacher_u_preds)

        # total loss
        cons_w = self._consistency_weight()
        total_loss = supervised_weight * sup_loss + cons_w * cons_loss

        # backward & step for student
        optimizer.zero_grad()
        total_loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(self.student.parameters(), clip_grad)
        optimizer.step()

        # update ema teacher
        self.update_ema()

        # increment counters
        self._global_step += 1

        return {
            "preds_l": preds_l,
            "supervised_loss": sup_loss.detach().cpu(),
            "consistency_loss": cons_loss.detach().cpu(),
            "total_loss": total_loss.detach().cpu(),
        }

    # def predict_student(self, imgs: torch.Tensor) -> torch.Tensor:
    #     self.student.eval()
    #     imgs = imgs.to(self.device)
    #     with torch.no_grad():
    #         return self.student(imgs)

    # def predict_teacher(self, imgs: torch.Tensor) -> torch.Tensor:
    #     self.teacher.eval()
    #     imgs = imgs.to(self.device)
    #     with torch.no_grad():
    #         return self.teacher(imgs)

    def set_epoch(self, epoch: int):
        """Set current epoch (used for rampup schedule)."""
        self._current_epoch = epoch

# --------------------- Example usage snippet ---------------------

if __name__ == "__main__":
    wrapper = MeanTeacherWrapper(BM())
    imgs_l = [torch.randn((1,3,224,224)).cuda(), torch.randn((1,3,224,224)).cuda()]
    imgs_u = [torch.randn((1,3,224,224)).cuda(), torch.randn((1,3,224,224)).cuda()]
    optimizer = optim.Adam(wrapper.parameters(), lr=5e-6, weight_decay=1e-4)
    targets = torch.randn((1,1,28,28)).cuda()
    bayesian_loss = F.mse_loss

    wrapper.set_epoch(100)
    metrics = wrapper.training_step((imgs_l, targets), imgs_u, optimizer, bayesian_loss)
    print(metrics["preds_l"].shape)
