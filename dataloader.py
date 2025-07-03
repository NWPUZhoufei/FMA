from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

# Data transformation with augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

# Load datasets
def load_data(data_root, txt_path, phase, batch_size=128,  num_workers=4, shuffle=True):
    
    print('Loading data from %s' % (txt_path))

    if phase=='train':
        transform = data_transforms['train']
    
    if phase=='val':
        transform = data_transforms['val']
    
    if phase=='test':
        transform = data_transforms['test']        
        
    print('Use data transformation:', transform)

    dataset = LT_Dataset(data_root, txt_path, transform)
    
    dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=None, num_workers=num_workers)
    
    return dataLoader


    
    
