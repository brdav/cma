import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from models.backbones import MixVisionTransformer, ResNet
from models.heads import DeepLabV2Head, SegFormerHead
from PIL import Image
from tqdm import tqdm


@torch.inference_mode()
def generate_pseudo_labels(model, datamodule):
    print("Generating pseudo labels...")

    datamodule.prepare_data()
    datamodule.setup('predict')
    assert len(datamodule.predict_ds) == 1

    # iterate through all samples to collect statistics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    model.use_ensemble_inference = True
    model.inference_ensemble_scale = [
        0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    print("Collecting statistics...")
    conf_dict = defaultdict(list)
    for batch in tqdm(datamodule.predict_dataloader()[0]):
        batch = model.transfer_batch_to_device(
            batch, device, dataloader_idx=0)
        orig_size = datamodule.predict_ds[0].orig_dims
        x = batch['image'] if 'image' in batch.keys() else batch['image_ref']
        y_hat = model.forward(x, orig_size)
        probs = y_hat if model.use_ensemble_inference else nn.functional.softmax(y_hat, dim=1)
        pred_prob, pred_labels = probs.max(dim=1)
        for idx_cls in range(model.head.num_classes):
            idx_temp = pred_labels == idx_cls
            if idx_temp.any():
                conf_cls_temp = pred_prob[idx_temp][::10].cpu(
                ).tolist()
                conf_dict[idx_cls].extend(conf_cls_temp)

    trg_portion = 0.2
    cls_thresh = torch.ones(
        model.head.num_classes, dtype=torch.float, device=device)
    for idx_cls in range(model.head.num_classes):
        cls_probs = sorted(conf_dict[idx_cls], reverse=True)
        len_cls_sel = int(len(cls_probs) * trg_portion)
        if len_cls_sel > 0:
            cls_thresh[idx_cls] = cls_probs[len_cls_sel - 1]

    print("Extract PL...")
    dataset_name = datamodule.predict_on[0]
    if isinstance(model.backbone, MixVisionTransformer) and isinstance(model.head, SegFormerHead):
        model_name = 'segformer'
    elif isinstance(model.backbone, ResNet) and isinstance(model.head, DeepLabV2Head):
        model_name = 'deeplabv2'
    else:
        raise RuntimeError('unknown model configuration')
    save_dir = os.path.join(
        os.environ['$DATA_DIR'], 'pseudo_labels', 'pseudo_labels_train_{}_cma_{}'.format(dataset_name, model_name))
    os.makedirs(save_dir, exist_ok=True)

    # iterate through all samples to generate pseudo label
    for batch in tqdm(datamodule.predict_dataloader()[0]):
        filename = batch['filename']
        if all(os.path.isfile(os.path.join(
                save_dir, '.'.join([*im_name.split('.')[:-1], 'png']))) for im_name in filename):
            continue
        batch = model.transfer_batch_to_device(
            batch, device, dataloader_idx=0)
        orig_size = datamodule.predict_ds[0].orig_dims
        x = batch['image'] if 'image' in batch.keys() else batch['image_ref']
        y_hat = model.forward(x, orig_size)
        probs = y_hat if model.use_ensemble_inference else nn.functional.softmax(y_hat, dim=1)
        weighted_prob = probs / cls_thresh.view(1, -1, 1, 1)
        weighted_conf, preds = weighted_prob.max(dim=1)
        preds[weighted_conf < 1] = 255

        for pred, im_name in zip(preds, filename):
            arr = pred.cpu().numpy()
            image = Image.fromarray(arr.astype(np.uint8))
            image.save(os.path.join(
                save_dir, '.'.join([*im_name.split('.')[:-1], 'png'])))
