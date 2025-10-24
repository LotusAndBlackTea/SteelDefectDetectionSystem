
#cuda=12.9
#pytorch=2.8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F

# 配置参数
class Config:
    # 模型配置
    model_name = 'unet'
    num_classes = 4
    input_size = (256, 256)
    
    # 训练配置
    batch_size = 16

    epochs = 1000
    learning_rate = 0.001
    weight_decay = 0.0005
    
    # 数据配置
    train_csv_path = 'dataset/train.csv'
    train_image_dir = 'dataset/train_images'
    preprocessed_path = 'dataset/preprocessed_data.pkl'
    
    # 保存路径
    checkpoint_dir = 'checkpoints'
    best_model_path = 'best_steel_defect_model.pth'
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
config = Config()

# 数据增强
def get_train_transforms():
    return A.Compose([
        A.Resize(height=config.input_size[0], width=config.input_size[1]),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=config.input_size[0], width=config.input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# 数据集类
class SteelDefectDataset(Dataset):
    def __init__(self, image_dir, df, transform=None, image_size=(256, 256), use_cache=True):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.image_size = image_size
        self.image_ids = df['ImageId'].unique()
        self.use_cache = use_cache
        self.cache = {}
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
            
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        
        # 读取图像
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建mask (4个类别)
        mask = np.zeros((*self.image_size, 4), dtype=np.float32)
        
        # 获取该图像的所有标注
        image_df = self.df[self.df['ImageId'] == image_id]
        
        for _, row in image_df.iterrows():
            class_id = int(row['ClassId']) - 1
            if pd.notna(row['EncodedPixels']):
                class_mask = self.rle_decode(row['EncodedPixels'], (256, 1600))
                class_mask = cv2.resize(class_mask, (self.image_size[1], self.image_size[0]))
                mask[:, :, class_id] = class_mask
        
        # 调整图像大小
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.dim() == 3 and mask.shape[0] != config.num_classes:
                mask = mask.permute(2, 0, 1)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
            mask = torch.from_numpy(mask.transpose(2, 0, 1).astype(np.float32))
        
        result = (image, mask)
        if self.use_cache:
            self.cache[idx] = result
            
        return result
    
    def rle_decode(self, mask_rle, shape):
        if pd.isna(mask_rle):
            return np.zeros(shape, dtype=np.uint8)
            
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

# 修正的UNet - 修复通道不匹配问题
class FixedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[32, 64, 128, 256]):
        super(FixedUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder
        for feature in features:
            self.encoder.append(self._block(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = self._block(features[-1], features[-1] * 2)
        
        # Decoder - 修正通道数
        for i, feature in enumerate(reversed(features)):
            # 转置卷积上采样
            if i == 0:
                # 第一个解码块：从bottleneck到最大特征
                self.decoder.append(
                    nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2)
                )
                self.decoder.append(self._block(features[-1] * 2, features[-1]))
            else:
                # 后续解码块
                prev_feature = features[len(features) - i]
                self.decoder.append(
                    nn.ConvTranspose2d(prev_feature * 2, feature, kernel_size=2, stride=2)
                )
                self.decoder.append(self._block(feature * 2, feature))
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Reverse skip connections
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # 确保尺寸匹配
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            
            # 拼接跳跃连接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
        
        return self.final_conv(x)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 更简单的UNet实现 - 避免复杂的通道计算
class EfficientUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, features=[16, 32, 64, 128]):
        super(EfficientUNet, self).__init__()
        
        # Encoder - 使用更少的特征通道
        self.enc1 = self._efficient_block(in_channels, features[0])
        self.enc2 = self._efficient_block(features[0], features[1])
        self.enc3 = self._efficient_block(features[1], features[2])
        self.enc4 = self._efficient_block(features[2], features[3])
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._efficient_block(features[3], features[3] * 2)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features[3] * 2, features[3], kernel_size=2, stride=2)
        self.dec4 = self._efficient_block(features[3] * 2, features[3])
        
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = self._efficient_block(features[2] * 2, features[2])
        
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = self._efficient_block(features[1] * 2, features[1])
        
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = self._efficient_block(features[0] * 2, features[0])
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)
    
    def _efficient_block(self, in_channels, out_channels):
        """更高效的卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# 损失函数
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

# 学习率调度器
class WarmupCosineAnnealingLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2 
                    for base_lr in self.base_lrs]

# 训练函数
def train_model():
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 加载预处理数据
    with open(config.preprocessed_path, 'rb') as f:
        data_config = pickle.load(f)
    
    # 创建数据集
    train_dataset = SteelDefectDataset(
        config.train_image_dir,
        data_config['train_df'],
        transform=get_train_transforms(),
        image_size=config.input_size,
        use_cache=False  # 暂时关闭缓存避免问题
    )
    
    val_dataset = SteelDefectDataset(
        config.train_image_dir,
        data_config['val_df'],
        transform=get_val_transforms(),
        image_size=config.input_size,
        use_cache=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows上设置为0
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 初始化模型 - 使用简单的UNet
    model = EfficientUNet(in_channels=3, out_channels=config.num_classes)
    model = model.to(config.device)
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = WarmupCosineAnnealingLR(
        optimizer, 
        warmup_epochs=5, 
        total_epochs=config.epochs,
        eta_min=config.learning_rate * 0.01
    )
    
    # 损失函数
    criterion = CombinedLoss(alpha=0.7)
    
    # 训练历史
    train_losses = []
    val_losses = []
    learning_rates = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("开始训练...")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"使用设备: {config.device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(config.device)
            masks = masks.to(config.device)
            
            # 确保mask维度正确
            if masks.dim() == 4 and masks.size(1) != config.num_classes:
                if masks.size(-1) == config.num_classes:
                    masks = masks.permute(0, 3, 1, 2)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(config.device)
                masks = masks.to(config.device)
                
                if masks.dim() == 4 and masks.size(1) != config.num_classes:
                    if masks.size(-1) == config.num_classes:
                        masks = masks.permute(0, 3, 1, 2)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {learning_rates[-1]:.6f}')
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
                'config': config
            }, config.best_model_path)
            print(f'最佳模型已保存，验证损失: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停: 验证损失在 {patience} 个epoch内没有提升')
                break
    
    # 绘制训练曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, label='学习率')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('学习率变化')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return model

# 评估函数
def evaluate_model(model, data_loader):
    model.eval()
    total_dice = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc='评估中'):
            images = images.to(config.device)
            masks = masks.to(config.device)
            
            if masks.dim() == 4 and masks.size(1) != config.num_classes:
                if masks.size(-1) == config.num_classes:
                    masks = masks.permute(0, 3, 1, 2)
            
            outputs = model(images)
            
            # 计算Dice系数
            dice_loss = DiceLoss()
            dice_score = 1 - dice_loss(outputs, masks)
            total_dice += dice_score.item() * images.size(0)
            total_samples += images.size(0)
    
    avg_dice = total_dice / total_samples
    print(f'平均Dice系数: {avg_dice:.4f}')
    return avg_dice

# 预测函数
def predict_single_image(model, image_path):
    model.eval()
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_val_transforms()
    transformed = transform(image=image)
    input_tensor = transformed['image'].unsqueeze(0).to(config.device)
    
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)
    
    return output.cpu().numpy()

# 主函数
def main():
    print("开始训练钢材表面缺陷检测模型...")
    print(f"使用设备: {config.device}")
    print(f"模型: SimpleUNet")
    print(f"输入尺寸: {config.input_size}")
    print(f"批大小: {config.batch_size}")
    print(f"类别数: {config.num_classes}")
    
    # 训练模型
    model = train_model()
    
    # 加载最佳模型进行评估
    checkpoint = torch.load(config.best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 重新创建验证集进行评估
    with open(config.preprocessed_path, 'rb') as f:
        data_config = pickle.load(f)
    
    val_dataset = SteelDefectDataset(
        config.train_image_dir,
        data_config['val_df'],
        transform=get_val_transforms(),
        image_size=config.input_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # 评估模型
    print("开始评估模型...")
    dice_score = evaluate_model(model, val_loader)
    
    print("\n训练完成!")
    print(f"最终Dice系数: {dice_score:.4f}")

if __name__ == "__main__":
    main()