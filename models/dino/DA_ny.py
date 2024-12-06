import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
from torch.autograd import Function



def mask_to_box_with_categories(boxes, labels, size, image_size, num_classes):
    """
    Generate category masks based on the given bounding boxes and labels, and
    return the categories that actually exist。

    Parameters:
    - boxes: Tensor，bounding box coordinates
    - labels: Tensor，category labels
    - size: Original image size
    - image_size: Target size (size of the mask)
    - num_classes: Total number of categories

    Returns:
    - masks: Tensor, category masks with shape [num_classes, H, W]
    - present_classes: list, the categories that actually exist in the image
    """
    # Initialize the mask dictionary and the list of existing categories
    masks_dict = {label: torch.zeros((1, image_size[0], image_size[1]), dtype=torch.float32).cuda() for label in range(num_classes)}
    present_classes = set()

    # Check if boxes and labels are empty
    if boxes is None or len(boxes) == 0:
        return torch.zeros((num_classes, image_size[0], image_size[1]), dtype=torch.float32).cuda(), []

    img_h, img_w = size[1], size[0]
    scale_fct = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).cuda()

    # Ensure that the dimensions of boxes and labels are correct
    if len(boxes.size()) != 3:
        boxes = boxes.unsqueeze(0)
        labels = labels.unsqueeze(0)

    boxes = boxes * scale_fct
    boxes = boxes[0]
    labels = labels[0]

    for box, label in zip(boxes, labels):
        x_center, y_center, width, height = box
        xmin = int((x_center - width / 2).item())
        xmax = int((x_center + width / 2).item())
        ymin = int((y_center - height / 2).item())
        ymax = int((y_center + height / 2).item())
        # Debug usage
        # if xmin < 0 or ymin < 0 or xmax > image_size[1] or ymax > image_size[0]:
        #     print(f"Warning: Box coordinates are out of image bounds: ({xmin}, {ymin}, {xmax}, {ymax})")
        #     continue

        label_int = int(label.item())
        if label_int in masks_dict:
            masks_dict[label_int][0, ymin:ymax, xmin:xmax] = 1
            present_classes.add(label_int)  # Add the categories that actually exist

    # Convert the mask to a tensor
    masks = torch.cat([masks_dict[label] for label in range(num_classes)], dim=0)


    # Return the list of categories that actually exist for subsequent loss calculation
    return masks, list(present_classes)  #



def downsample_and_apply_masks(masks, srcs):
    """
    Downsample the mask for each category and then apply it to the feature map output by each backbone.

    Parameters:
    - masks: A list containing the category masks for each image, where each mask has the shape [num_classes, H, W]
    - srcs: A list of feature maps output by the backbone, where each feature map has the shape [B, C, H_feature, W_feature]

    Returns:
    - masked_srcs: A list containing the masked feature maps, where each feature map has the same shape as the original feature map
    """

    B, num_channels, _, _ = srcs[0].shape  # Get the batch size and number of channels of the feature map
    num_classes = masks[0].shape[0]  # Get the number of Categories

    # Initialize the list to store the feature maps after masking
    masked_srcs = [[] for _ in range(len(srcs))]  # Used to store the masked feature maps for each feature layer

    # For each feature map layer
    for idx, src in enumerate(srcs):
        feature_size = src.shape[-2:]  # Obtain the size of the feature map (H_feature, W_feature)

        # Process each category for each mask individually
        for mask in masks:
            if not isinstance(mask, torch.Tensor):
                raise ValueError("The mask is not a tensor. Please check the input data.")

            mask = mask.to(src.device)  # Move the mask to the same device as the feature map

            # Downsampling
            downsampled_masks = F.interpolate(mask.unsqueeze(1).float(), size=feature_size, mode='nearest').squeeze(1)  # [num_classes, H_feature, W_feature]

            # Expand the mask to match the channel size and batch size of the feature map
            expanded_masks = downsampled_masks.unsqueeze(1).expand(-1, num_channels, -1, -1)  # [num_classes, C, H_feature, W_feature]

            #Apply the mask to the feature map
            for b in range(B):
                combined_mask = expanded_masks.sum(dim=0)  #  [C, H_feature, W_feature]
                masked_src = src[b] * combined_mask  # Feature map after masking [C, H_feature, W_feature]
                masked_srcs[idx].append(masked_src.unsqueeze(0))  # Add to the category list

    # Merge the feature maps after masking
    final_masked_srcs = [torch.cat(masked, dim=0) for masked in masked_srcs]

    return final_masked_srcs






def class_mask_features(class_masks, srcs):
    """
    Perform masking operations on different categories in the feature maps.

    Parameters:
    - class_masks: List of masks, each with shape [N, 1, H, W].
    - srcs: List of feature maps, each with shape [B, C, H, W].

    Returns:
    - masked_class_features: List of feature maps after masking.
    """

    def apply_mask(mask, feature):
        masked_features = []
        for f in feature:
            # Initialize a mask list for each feature map
            resized_masks = []
            for m in mask:
                # Downsampling
                downsampled_mask = F.interpolate(m.float(), size=f.shape[-2:], mode='nearest')
                expanded_mask = downsampled_mask.expand(-1, f.size(1), -1, -1)
                resized_masks.append(expanded_mask)

            # Concatenate all masks along the batch dimension
            combined_mask = torch.cat(resized_masks, dim=0)

            # Element-wise multiplication
            masked_feature = f * combined_mask
            masked_features.append(masked_feature)
        return masked_features

    masked_class_features = apply_mask(class_masks, srcs)
    return masked_class_features


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha = 0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output


# GRLayer is applied to the input x
def grad_reverse(x):
    return GRLayer.apply(x)

# Domain Discriminator
# A convolutional neural network is defined with two convolutional layers (Conv1 and Conv2) and a ReLU activation function
class DA_discriminator(nn.Module):
    def __init__(self, dim):
        super(DA_discriminator, self).__init__()
        self.dim = dim  # feat layer          256*H*W for vgg16
        self.Conv1 = nn.Conv2d(self.dim, 512, kernel_size=1, stride=1, bias=False)
        self.Conv2 = nn.Conv2d(512, 2, kernel_size=1, stride=1, bias=False)
        self.reLu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = grad_reverse(x)
        x = self.reLu(self.Conv1(x))
        x = self.Conv2(x)
        return x
