from torchvision import datasets
datasets.CIFAR10(root='./data', train=True, download=True)
datasets.MNIST(root='./data', train=True, download=True)
