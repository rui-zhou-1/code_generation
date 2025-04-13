class Model(nn.Module):
    def __init__(self, input_dim, num_classes):
        
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        
        out = self.fc1(x)
        return out

if __name__ == '__main__':
    
    input_dim = 5
    num_classes = 2
    
    model = Model(input_dim, num_classes)
    
    print(model)
    
    inputs = torch.randn(1, input_dim)
    
    out = model(inputs)
    
    print(out)