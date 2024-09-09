import torch

kernel = 7
hidden_channels = 128
num_layers = 7
grid_length = 5000
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.leaky_relu= torch.nn.LeakyReLU(negative_slope=0.1)

        self.conv_in = torch.nn.Conv1d(1, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')

        self.conv2 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')
        self.conv3 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')
        self.conv4 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')
        self.conv5 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')
        self.conv6 = torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size = kernel, padding='same', padding_mode='replicate')

        self.conv_out = torch.nn.Conv1d(hidden_channels, 1, kernel_size = kernel, padding='same', padding_mode='replicate')

    def resnet(self, x):

      x = self.leaky_relu(self.conv_in(x))

      x = self.leaky_relu(self.conv2(x))
      x = self.leaky_relu(self.conv3(x))
      x = self.leaky_relu(self.conv4(x))
      x = self.leaky_relu(self.conv5(x))
      x = self.leaky_relu(self.conv6(x))

      x = self.leaky_relu(self.conv_out(x))

      return x

    def forward(self, x):

      # Apply Data-driven, nonlinear NN filter with consistency
      x_nn = x + self.resnet(x)
      ones_vec = torch.ones(1,1,grid_length, device='cpu')
      consistency = ones_vec + self.resnet(ones_vec)
      constant = sum(consistency.view(-1))/grid_length

      x_nn = x_nn/constant

      return x_nn


def load_nn_filter(nn_filename):
    MLfilter= Net()
    MLfilter.load_state_dict(torch.load(nn_filename, map_location=torch.device('cpu')))
    MLfilter.eval()

    return MLfilter