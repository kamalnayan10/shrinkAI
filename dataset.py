from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

class DIV2KDataset(Dataset):
    def __init__(self, root_dir, crop_size=256):
        
        self.image_paths = sorted([
            os.path.join(root_dir, f) for f in os.listdir(root_dir)
        ])

        self.transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

if __name__ == "__main__":
    
    train_dataset = DIV2KDataset("DIV2K_train_HR")
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    images = next(iter(train_loader))
    print(images.shape)