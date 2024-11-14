import torch
import torch.nn as nn
import torch.nn.init as init

class LogisticRegression(nn.Module):
    def __init__(self, d=10):
        """ Logisitic Regression Meta Model """
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(d, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = self.linear(X)
        X = self.sigmoid(X)
        return X

   
class LinearRegressionModulate(nn.Module):
    def __init__(self, d=10):
        super(LinearRegressionModulate, self).__init__()
        self.linear = nn.Linear(d, 1, bias=True)    
        self.relu = nn.ReLU()

    def forward(self, X, orig_w):
        X = self.linear(X)
        # X = self.sigmoid(X)
        out = X + orig_w.unsqueeze(-1)
        out = self.relu(out) + 1e-7 # for numerical stability
        out = out / torch.sum(out)
        out *= torch.sum(orig_w)
        return out
    

class LogisticRegressionModulate(nn.Module):
    def __init__(self, d=10):
        """ Logisitic Regression Meta Model """
        super(LogisticRegressionModulate, self).__init__()
        self.linear = nn.Linear(d, 3, bias=True)
        self.linear2 = nn.Linear(3, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, X , orig_w):
        X = self.linear(X)
        X = self.sigmoid(X)
        X = self.linear2(X)
        # X = self.sigmoid(X)
        out = X + orig_w.unsqueeze(-1)
        out = self.relu(out)
        out = out / torch.sum(out)
        out *= torch.sum(orig_w)
        return out
    
class LogisticRegression1(nn.Module):
    def __init__(self, d=10):
        """ Logisitic Regression Meta Model """
        super(LogisticRegression1, self).__init__()
        self.linear = nn.Linear(d-1, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, X):
        tmp = self.linear(X[:, :-1])
        tmp = self.tanh(tmp)
        out = tmp + X[:, -1].unsqueeze(-1) * 100
        return self.relu(out)

# class TwoLayerNN(nn.Module):
#     def __init__(self, d=2):
#         super(TwoLayerNN, self).__init__()
#         self.fc1 = nn.Linear(d, 3)
#         self.fc2 = nn.Linear(3, 2)
#         self.fc3 = nn.Linear(2, 1)
        
#         self.sigmoid = nn.Sigmoid()

#         # Initialize all parameters to zero
#         # for param in self.parameters():
#         #     param.data.fill_(0)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
    
# class TwoLayerNN(nn.Module):
#     def __init__(self, d=2):
#         super(TwoLayerNN, self).__init__()
#         self.fc1 = nn.Linear(d, 10)
#         self.fc2 = nn.Linear(10, 10)
#         self.fc3 = nn.Linear(10, 1)
        
#         self.sigmoid = nn.Sigmoid()

#         # Initialize all parameters to zero
#         # for param in self.parameters():
#         #     param.data.fill_(0)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
 
class TwoLayerNN(nn.Module):
    def __init__(self, d=2):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(d, 3)
        self.fc2 = nn.Linear(3, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Initialize all parameters to zero
        # for param in self.parameters():
        #     param.data.fill_(0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    
# class TwoLayerNNModulate(nn.Module):
#     def __init__(self, d=2):
#         super(TwoLayerNNModulate, self).__init__()
#         self.fc1 = nn.Linear(d, 3)
#         self.fc2 = nn.Linear(3, 2)
#         self.fc3 = nn.Linear(2, 1)
        
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, orig_w):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         # normalize the sum of weights
#         out = x * 0.2 + orig_w.unsqueeze(-1)
#         out /= torch.sum(out)
#         out *= torch.sum(orig_w)

#         return out

# class TwoLayerNNModulate(nn.Module):
#     def __init__(self, d=2):
#         super(TwoLayerNNModulate, self).__init__()
#         self.fc1 = nn.Linear(d, 7)
#         self.fc2 = nn.Linear(7, 5)
#         self.fc3 = nn.Linear(5, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, orig_w):
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         # normalize the sum of weights
#         out = x * 0.5 + orig_w.unsqueeze(-1)
#         out /= torch.sum(out)
#         out *= torch.sum(orig_w)

#         return out
    
# class TwoLayerNNModulate(nn.Module):
#     def __init__(self, d=2):
#         super(TwoLayerNNModulate, self).__init__()
#         self.fc1 = nn.Linear(d, 7)
#         self.fc2 = nn.Linear(7, 5)
#         self.fc3 = nn.Linear(5, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, orig_w):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         # normalize the sum of weights
#         out = x * 0.5 + orig_w.unsqueeze(-1)
#         out /= torch.sum(out)
#         out *= torch.sum(orig_w)
#         return out


class TwoLayerNNModulate(nn.Module):
    def __init__(self, d=2):
        super(TwoLayerNNModulate, self).__init__()
        self.fc1 = nn.Linear(d, 7)
        self.fc2 = nn.Linear(7, 7)
        self.fc3 = nn.Linear(7, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, orig_w):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        out = x + orig_w.unsqueeze(-1)
        out = self.relu(out) + 0.001
        out = out / torch.sum(out)
        out *= torch.sum(orig_w)

        return out


class BigTwoLayerNNModulate(nn.Module):
    def __init__(self, d=2):
        super(BigTwoLayerNNModulate, self).__init__()
        self.fc1 = nn.Linear(d, 7)
        self.fc2 = nn.Linear(7, 5)
        self.fc3 = nn.Linear(5, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, orig_w):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        # normalize the sum of weights
        out = x * 0.1 + orig_w.unsqueeze(-1)
        out = out / torch.sum(out)
        out *= torch.sum(orig_w)
        return out
    
class LinearRegression(nn.Module):
    def __init__(self, d=2, weight_init=None, bias_init=None):
        """ Logisitic Regression Meta Model """
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(d, 1, bias=True)
        self.relu = nn.ReLU()

        # Initialize the first weight and the bias to 0, and the second weight to 1
        with torch.no_grad():
            if weight_init is None:
                init_vec = torch.zeros(d)
                init_vec[-1] = 1.0
                self.linear.weight.copy_(init_vec)
            else:
                self.linear.weight.copy_(weight_init)
            
            if bias_init is None:
                self.linear.bias.fill_(0)
            else:
                self.linear.bias.fill_(bias_init)

    def forward(self, X):
        X = self.linear(X)
        X = self.relu(X)
        return X
    
class Monotonic(nn.Module):
    def __init__(self, I=3):
        super(Monotonic, self).__init__()
        
        self.num_transforms = I
        # Initializing the parameters
        self.d = nn.Parameter(torch.tensor([1.0]))
        self.a = nn.Parameter(torch.zeros(self.num_transforms))
        self.b = nn.Parameter(torch.zeros(self.num_transforms))
        self.c = nn.Parameter(torch.zeros(self.num_transforms))

         # Xavier initialization for a, b, c
        self.a = nn.Parameter(torch.tensor([0.1] * self.num_transforms))
        self.b = nn.Parameter(torch.tensor([0.1] * self.num_transforms))
        self.c = nn.Parameter(torch.tensor([0.1] * self.num_transforms))

    def forward(self, x):
        d_positive = torch.nn.functional.softplus(self.d)
        a_positive = torch.nn.functional.softplus(self.a)
        b_positive = torch.nn.functional.softplus(self.b)
        
        linear_term = x * d_positive
        tanh_terms = 0
        for i in range(self.num_transforms):
            tanh_terms += a_positive[i] * torch.tanh(b_positive[i] * (x + self.c[i]))

        
        
        return linear_term + tanh_terms