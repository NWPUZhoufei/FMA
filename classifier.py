import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, num_classes=1000):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = self.linear(x)
        return x
    
    

if __name__ == "__main__":
    
    linearClassifier = LinearClassifier(out_dim=738, num_classes=1000)
    x = torch.rand(2,738) 
    y = linearClassifier(x)
    print(y.size())


    

