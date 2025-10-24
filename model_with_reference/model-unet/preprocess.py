import os
import cv2
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def rle_decode(mask_rle, shape):
    """
    将RLE编码的解码为二值mask
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

class SteelDefectDataset(Dataset):
    def __init__(self, image_dir, df, transform=None, image_size=(256, 1600)):
        self.image_dir = image_dir
        self.df = df
        self.transform = transform
        self.image_size = image_size
        self.image_ids = df['ImageId'].unique()
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
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
            class_id = int(row['ClassId']) - 1  # 类别从0开始
            if pd.notna(row['EncodedPixels']):
                # 解码RLE
                class_mask = rle_decode(row['EncodedPixels'], self.image_size)
                mask[:, :, class_id] = class_mask
        
        # 调整图像和mask大小
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            for i in range(4):
                mask_channel = cv2.resize(mask[:, :, i], (self.image_size[1], self.image_size[0]))
                mask[:, :, i] = mask_channel
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换维度格式 [H, W, C] -> [C, H, W]
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        mask = mask.transpose(2, 0, 1).astype(np.float32)
        
        return {
            'image': torch.tensor(image, dtype=torch.float),
            'mask': torch.tensor(mask, dtype=torch.float),
            'image_id': image_id
        }

def prepare_data(config):
    """
    数据预处理主函数
    """
    # 读取标注文件
    df = pd.read_csv(config['train_csv_path'])
    
    # 分离ImageId和ClassId
    df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
    
    print(f"总样本数: {len(df)}")
    print(f"唯一图像数: {df['ImageId'].nunique()}")
    print(f"缺陷分布:\n{df['ClassId'].value_counts().sort_index()}")
    
    # 检查图像文件是否存在
    train_image_dir = config['train_image_dir']
    available_images = set(os.listdir(train_image_dir))
    df = df[df['ImageId'].isin(available_images)]
    print(f"有效图像数: {df['ImageId'].nunique()}")
    
    # 分割训练集和验证集
    image_ids = df['ImageId'].unique()
    train_ids, val_ids = train_test_split(
        image_ids, 
        test_size=config['val_ratio'], 
        random_state=config['seed']
    )
    
    train_df = df[df['ImageId'].isin(train_ids)]
    val_df = df[df['ImageId'].isin(val_ids)]
    
    print(f"训练集图像: {len(train_ids)}")
    print(f"验证集图像: {len(val_ids)}")
    
    # 创建数据集配置
    data_config = {
        'image_size': config['image_size'],
        'num_classes': 4,
        'class_names': ['Class1', 'Class2', 'Class3', 'Class4'],
        'train_ids': train_ids.tolist(),
        'val_ids': val_ids.tolist(),
        'train_df': train_df,
        'val_df': val_df
    }
    
    # 保存预处理数据
    with open(config['output_path'], 'wb') as f:
        pickle.dump(data_config, f)
    
    print(f"数据预处理完成! 保存至: {config['output_path']}")
    
    return data_config

if __name__ == "__main__":
    # 配置参数
    config = {
        'train_csv_path': 'dataset/train.csv',
        'train_image_dir': 'dataset/train_images',
        'test_image_dir': 'dataset/test_images',
        'output_path': 'dataset/preprocessed_data.pkl',
        'image_size': (256, 1600),  # (height, width)
        'val_ratio': 0.2,
        'seed': 42
    }
    
    # 运行数据预处理
    data_config = prepare_data(config)
    
    # 显示数据统计信息
    print("\n数据统计:")
    print(f"图像尺寸: {data_config['image_size']}")
    print(f"类别数量: {data_config['num_classes']}")
    print(f"训练集大小: {len(data_config['train_ids'])}")
    print(f"验证集大小: {len(data_config['val_ids'])}")