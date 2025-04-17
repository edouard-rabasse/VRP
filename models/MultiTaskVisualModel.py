from torchvision import transforms
from src.data_loader_mask import CustomDataset

transform = transforms.Compose([
    transforms.ToPILImage(),        # Convert numpy array (from cv2) to PIL Image
    transforms.Resize((84, 84)),      # Resize both image and mask to 84x84
    transforms.ToTensor(),            # Convert to tensor (scales to [0,1])
])


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class MultiTaskVisualScoringModel(nn.Module):
    def __init__(self, input_shape=(3, 84, 84)):
        """
        Args:
            input_shape: The shape of the input image. For example, (3, 84, 84)
        """
        super(MultiTaskVisualScoringModel, self).__init__()
        # Shared encoder (CNN backbone)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Compute spatial dimensions after the convolutional layers:
        # For an 84x84 input:
        # conv1: output size = ((84 - 8) // 4) + 1 = 20  → shape: [N, 32, 20, 20]
        # conv2: output size = ((20 - 4) // 2) + 1 = 9   → shape: [N, 64, 9, 9]
        # conv3: output size = ((9 - 3) // 1) + 1 = 7    → shape: [N, 64, 7, 7]
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self._forward_conv(dummy_input)
            self.flattened_size = out.view(1, -1).shape[1]
        
        # Classification head
        self.fc1 = nn.Linear(self.flattened_size, 1024)
        self.fc_clf = nn.Linear(1024, 2)  # 2 outputs: original (0) or modified (1)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        
        # Segmentation head: here we use simple upsampling
        # We take the conv3 feature map (shape [N, 64, 7, 7]) and upsample to the desired segmentation size (84x84)
        # size of the segmentation
        self.segmentation_head = nn.Sequential(
            nn.Upsample(size=(input_shape[1],input_shape[2]), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)  # output 1-channel mask
        )
        
        # Initialization (using Kaiming for Conv and Linear layers)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                
    def _forward_conv(self, x):
        # Shared encoder forward path
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        return x
    
    def forward(self, x):
        # Compute shared features
        features = self._forward_conv(x)  # shape: [N, 64, 7, 7]
        # Classification branch
        clf = features.view(features.size(0), -1)
        clf = F.leaky_relu(self.fc1(clf), negative_slope=0.01)
        clf_logits = self.fc_clf(clf)
        # Segmentation branch
        seg_logits = self.segmentation_head(features)  # shape: [N, 1, 84, 84]
        return clf_logits, seg_logits


import torch
import torch.nn as nn

def train_model_multi_task(model, train_loader, test_loader, num_epochs, device, learning_rate, lambda_seg=1.0):
    """
    Train the multi-task model.
    Args:
        model: the multi-task model.
        train_loader: DataLoader for training.
        test_loader: DataLoader for testing (evaluation).
        num_epochs: Number of epochs to train.
        device: 'cpu' or 'cuda'.
        learning_rate: learning rate for optimizer.
        lambda_seg: weight for the segmentation loss.
    Returns:
        model, optimizer, scheduler after training.
    """
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Loss functions: classification and segmentation
    criterion_cls = nn.CrossEntropyLoss()        # expects (logits, target labels)
    criterion_seg = nn.BCEWithLogitsLoss()         # for pixel-wise mask prediction
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images = images.to(device)               # shape: [N, 3, H, W] (e.g., 84x84)
            labels = labels.to(device)               # shape: [N]
            masks = masks.to(device).float()         # shape: [N, 1, H, W]
            
            optimizer.zero_grad()
            
            clf_logits, seg_logits = model(images)
            loss_cls = criterion_cls(clf_logits, labels)
            loss_seg = criterion_seg(seg_logits, masks)
            loss = loss_cls + lambda_seg * loss_seg
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            running_cls_loss += loss_cls.item() * images.size(0)
            running_seg_loss += loss_seg.item() * images.size(0)
            preds = torch.argmax(clf_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        scheduler.step()
        epoch_loss = running_loss / total
        epoch_cls_loss = running_cls_loss / total
        epoch_seg_loss = running_seg_loss / total
        epoch_acc = correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}: Total Loss: {epoch_loss:.4f} " 
              f"(Cls: {epoch_cls_loss:.4f}, Seg: {epoch_seg_loss:.4f}), Acc: {epoch_acc*100:.2f}%")
    # evaluate on test set
    test_cls_loss = 0.0
    test_seg_loss = 0.0
    test_loss = 0.0
    correct = 0
    total = 0
    for images, labels, masks in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device).float()
        
        with torch.no_grad():
            clf_logits, seg_logits = model(images)
            loss_cls = criterion_cls(clf_logits, labels)
            loss_seg = criterion_seg(seg_logits, masks)
            loss = loss_cls + lambda_seg * loss_seg
        
        preds = torch.argmax(clf_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        test_loss = loss.item() * images.size(0)
        test_cls_loss = loss_cls.item() * images.size(0)
        test_seg_loss = loss_seg.item() * images.size(0)
    print(f"Test Loss: {test_loss:.4f} (Cls: {test_cls_loss:.4f}, Seg: {test_seg_loss:.4f}), Acc: {correct/total*100:.2f}%")


    return model, optimizer, scheduler

if __name__ == "__main__":
    # Assuming your directory paths are defined:
    original_path, modified_path, mask_path = "MSH/MSH/plots/configuration1", "MSH/MSH/plots/configuration5", "data/MSH/mask5"

    from src.transform import image_transform_train, image_transform_test, mask_transform

    from src.data_loader_mask import load_data_mask
    train_loader, test_loader = load_data_mask(original_path, modified_path, batch_size=2, 
                                                train_ratio=0.8, image_size=(84,84), num_workers=4, mask_path=mask_path, num_max=20, image_transform_test=image_transform_test(size=(84,84)), image_transform_train=image_transform_train(size=(84,84)), mask_transform_train=mask_transform(size=(84,84)))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiTaskVisualScoringModel(input_shape=(3,84,84))
    model.to(device)

    # Train for, say, 10 epochs and use a lambda weight of 1.0 for segmentation loss.
    model, optimizer, scheduler = train_model_multi_task(model, train_loader, test_loader,
                                                num_epochs=10,
                                                device=device,
                                                learning_rate=1e-3,
                                                lambda_seg=1)
    # print results
    print("Training completed.")

    # test the model on test set and save an image of the overlay

