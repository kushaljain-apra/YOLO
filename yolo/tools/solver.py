import torch

from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format, convert_tensor_to_python
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, recall_score, precision_score

class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)
        self.conf_thresholds = [cfg.task.nms.min_confidence] * cfg.dataset.class_num
    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model
        self.predicts_json = []
        self.targets_json = []

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )

        for predict in predicts:
            self.predicts_json.append(convert_tensor_to_python(to_metrics_format(predict)))

        for target in targets:
            self.targets_json.append(convert_tensor_to_python(to_metrics_format(target)))

        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()

        base_path = Path(self.cfg.out_path, self.cfg.task.task)
        save_path = base_path / self.cfg.name
        plots_save_path = save_path / "plots"
        os.makedirs(plots_save_path, exist_ok=True)
        
        list_classes = self.cfg.dataset.class_list
        num_classes = self.cfg.dataset.class_num

        y_true = []
        y_pred = []
        confidences = []

        for i in range(len(self.predicts_json)):
            # Get ground truth and predictions for this image (empty lists if none exist)
            gt_annotations = self.targets_json[i]
            pred_annotations = self.predicts_json[i]

            # If there are no ground truth annotations and no predictions, skip this image
            if not gt_annotations and not pred_annotations:
                continue
            
            gt_classes = []
            pred_classes = []
            pred_confidences = []

            # Add ground truth classes
            for key, value in gt_annotations.items():
                if key == 'labels':
                    gt_classes.extend(value)

            for key, value in pred_annotations.items():
                if key == 'labels':
                    pred_classes.extend(value)
                if key == 'scores':
                    pred_confidences.extend(value)

            # If we have ground truth but no predictions, count as missed detections
            if gt_classes and not pred_classes:
                # print(f"WARN : No Preds!")
                y_true.extend(gt_classes)
                y_pred.extend([num_classes] * len(gt_classes))  # Assuming 0 is a valid class ID
                confidences.extend([0.0] * len(gt_classes))
            # If we have predictions but no ground truth, count as false positives
            elif pred_classes and not gt_classes:
                # print(f"WARN : No GT found!")
                y_true.extend([num_classes] * len(pred_classes))  # Assuming 0 is a valid class ID
                y_pred.extend(pred_classes)
                confidences.extend(pred_confidences)
            # If we have both, add them normally
            else:
                # print(f"NORMAL")
                if len(gt_classes) > len(pred_classes):
                    diff_len = len(gt_classes) - len(pred_classes)
                    pred_classes.extend([num_classes] * diff_len)
                elif len(pred_classes) > len(gt_classes):
                    diff_len = len(pred_classes) - len(gt_classes)
                    gt_classes.extend([num_classes] * diff_len)
                assert len(gt_classes) == len(pred_classes)
                y_true.extend(gt_classes)
                y_pred.extend(pred_classes)
                confidences.extend(pred_confidences)

        print(f"y_pred <{len(y_pred)}> y_true <{len(y_true)}>")
        assert len(y_pred) == len(y_true)

        valid_indices = [i for i in range(len(y_true)) if y_true[i] < num_classes and y_pred[i] < num_classes]
        y_true_filtered = [y_true[i] for i in valid_indices]
        y_pred_filtered = [y_pred[i] for i in valid_indices]
        confidences_filtered = [confidences[i] for i in valid_indices]

        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=list(range(num_classes)), normalize='true')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=".3f", cmap="Blues", xticklabels=list_classes, yticklabels=list_classes)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(plots_save_path, "confusion_matrix.png"))
        plt.close()

        f1_scores = f1_score(y_true_filtered, y_pred_filtered, zero_division=0, average=None, labels=list(range(num_classes)))
        plt.figure(figsize=(10, 5))
        plt.bar(list_classes, f1_scores, color="b")
        plt.xlabel("Class")
        plt.ylabel("F1-Score")
        plt.title("F1-Score per Class")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(plots_save_path, "f1_score.png"))
        plt.close()

        # Compute Precision-Recall Curve per Class (One-vs-Rest)
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            y_true_binary = [1 if label == i else 0 for label in y_true_filtered]
            y_pred_binary = [1 if label == i else 0 for label in y_pred_filtered]

            precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_binary)
            plt.plot(recall, precision, label=list_classes[i])

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall Curve (One-vs-Rest)")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plots_save_path, "precision_vs_recall.png"))
        plt.close()

        # Compute F1-Score vs Confidence per Class
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            y_true_binary = [1 if label == i else 0 for label in y_true_filtered]
            confidence_scores = [conf if y_pred_filtered[idx] == i else 0 for idx, conf in enumerate(confidences_filtered)]

            precision, recall, thresholds = precision_recall_curve(y_true_binary, confidence_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
            plt.plot(thresholds, f1_scores[:-1], label=list_classes[i])

        plt.xlabel("Confidence")
        plt.ylabel("F1-Score")
        plt.title("F1-Score vs Confidence")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plots_save_path, "f1_vs_confidence.png"))
        plt.close()

        # Compute Precision vs Confidence per Class
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            y_true_binary = [1 if label == i else 0 for label in y_true_filtered]
            confidence_scores = [conf if y_pred_filtered[idx] == i else 0 for idx, conf in enumerate(confidences_filtered)]

            precision, _, thresholds = precision_recall_curve(y_true_binary, confidence_scores)
            plt.plot(thresholds, precision[:-1], label=list_classes[i])

        plt.xlabel("Confidence")
        plt.ylabel("Precision")
        plt.title("Precision vs Confidence")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plots_save_path, "precision_vs_confidence.png"))
        plt.close()

        # Compute Recall vs Confidence per Class
        plt.figure(figsize=(10, 6))
        for i in range(num_classes):
            y_true_binary = [1 if label == i else 0 for label in y_true_filtered]
            confidence_scores = [conf if y_pred_filtered[idx] == i else 0 for idx, conf in enumerate(confidences_filtered)]

            _, recall, thresholds = precision_recall_curve(y_true_binary, confidence_scores)
            plt.plot(thresholds, recall[:-1], label=list_classes[i])

        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.title("Recall vs Confidence")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(plots_save_path, "recall_vs_confidence.png"))
        plt.close()

        # Precision per Class
        precision_scores = precision_score(y_true_filtered, y_pred_filtered, zero_division=0, average=None, labels=list(range(num_classes)))

        plt.figure(figsize=(10, 5))
        plt.bar(list_classes, precision_scores, color="g")
        plt.xlabel("Class")
        plt.ylabel("Precision Score")
        plt.title("Precision per Class")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(plots_save_path, "precision_per_class.png"))
        plt.close()

        # Recall per Class
        recall_scores = recall_score(y_true_filtered, y_pred_filtered, zero_division=0, average=None, labels=list(range(num_classes)))

        plt.figure(figsize=(10, 5))
        plt.bar(list_classes, recall_scores, color="c")
        plt.xlabel("Class")
        plt.ylabel("Recall Score")
        plt.title("Recall per Class")
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(plots_save_path, "recall_per_class.png"))
        plt.close()

        print(f"Plots saved in {plots_save_path}")

class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"])
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def on_train_epoch_end(self):
        """Save model at the end of each epoch."""
        save_dir = Path(self.trainer.default_root_dir) / "models"
        save_dir.mkdir(parents=True, exist_ok=True)

        epoch_num = self.current_epoch
        model_path = save_dir / f"epoch_{epoch_num}.pt"

        torch.save(self.model, model_path)
        print(f"💾 Model saved at {model_path}")

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
