class CustomDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.data = np.load(data_path)
        self.label = np.load(label_path)
        
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def __len__(self):
        return len(self.data)

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle, num_workers, pin_memory)
        
    def __iter__(self):
        for batch_data, batch_label in super(CustomDataLoader, self).__iter__():
            yield (batch_data, batch_label)

def get_dataloader(data_path, label_path, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CustomDataset(data_path, label_path)
    return CustomDataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)

def get_dataloader_from_path(data_path, label_path, batch_size=32, shuffle=True, num_workers=0, pin_memory=True):
    dataset = CustomDataset(data_path, label_path)
    return CustomDataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)

def get_dataloader_from_path_with_transform(data_path, label_path, transform, batch_size=32, shuffle=True, num_workers=0