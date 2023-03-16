from typing import Any, Literal, Optional

import torch
from torch import Tensor
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.utilities.compute import _safe_divide


def _my_jaccard_index_reduce(
    confmat: Tensor,
    average: Optional[Literal["macro", "none"]],
    over_present_classes: bool = False,
) -> Tensor:
    allowed_average = ["macro", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")
    confmat = confmat.float()
    if confmat.ndim == 3:  # multilabel
        raise NotImplementedError
    else:  # multiclass
        num = torch.diag(confmat)
        denom = confmat.sum(0) + confmat.sum(1) - num

        if over_present_classes:
            present_classes = confmat.sum(1) != 0
            num = torch.masked_select(num, present_classes)
            denom = torch.masked_select(denom, present_classes)

    jaccard = _safe_divide(num, denom)

    if average is None or average == "none":
        return jaccard
    return (jaccard / jaccard.numel()).sum()


class IoU(MulticlassJaccardIndex):

    def __init__(
        self,
        over_present_classes: bool = False,
        **kwargs: Any,
    ) -> None:
        self.over_present_classes = over_present_classes
        super().__init__(**kwargs)

    def compute(self) -> Tensor:
        return _my_jaccard_index_reduce(self.confmat, average=self.average, over_present_classes=self.over_present_classes)
