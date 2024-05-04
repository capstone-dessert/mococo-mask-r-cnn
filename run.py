import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score
import numpy as np
from dataset import load_data
import torch.optim as optim
from datetime import datetime
import time


def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets, idxs in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # Perform backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def box_iou(box1, box2):
    """Compute the intersection over union of two sets of boxes."""
    # Calculate the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area
    return iou


def evaluate(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    ious = []

    with torch.no_grad():
        for images, targets, idx in tqdm(data_loader):
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for target, output in zip(targets, outputs):
                # Calculate IoUs and update true/predicted labels for F1-score
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().detach().numpy()
                pred_scores = output['scores'].cpu().detach().numpy()
                pred_class = output['labels'].cpu().detach().numpy()

                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    max_iou = 0
                    selected_pred = None
                    for pred_box, pred_label in zip(pred_boxes, pred_class):
                        iou = box_iou(gt_box, pred_box)
                        if iou > max_iou:
                            max_iou = iou
                            selected_pred = pred_label
                    if max_iou >= 0.5:  # Considering an IoU threshold
                        ious.append(max_iou)
                        true_labels.append(gt_label)
                        pred_labels.append(selected_pred)
                    else:
                        true_labels.append(gt_label)
                        pred_labels.append(0)  # Assuming class 0 is background or no detection

    # Compute the F1-score for classification
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    mean_iou = np.mean(ious) if ious else 0
    print(f"Mean IoU: {mean_iou:.4f}, F1-Score: {f1:.4f}")

    return mean_iou, f1


def get_current_time():
    now = datetime.now()
    date = str(now.date())
    time = str(now.time()).split(":")
    return date + "-" + "".join(time[:2])


def save_model(epoch, model, optimizer, path="model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)

if __name__ == "__main__":
    num_classes = 26  # 1 class (background) + 25 classes
    model = get_model_instance_segmentation(num_classes)

    train_loader = load_data(train=True, batch_size=2, num_workers=1)
    valid_loader = load_data(train=False, batch_size=2, num_workers=1)

    # Create the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005,
                          momentum=0.9, weight_decay=0.0005)

    # StepLR scheduler example
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    model.to(device)

    num_epochs = 10
    current = get_current_time()

    for epoch in range(1, num_epochs + 1):
        print(f"epoch {epoch} - train")
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()  # Update learning rate

        print(f"epoch {epoch} - evaluate")
        mean_iou, f1 = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch} Evaluation - Mean IoU: {mean_iou:.4f}, F1-Score: {f1:.4f}\n")

        # Save model if performance improved
        save_model(epoch, model, optimizer, path=f"./save/{current}_model_epoch_{epoch}.pth")
