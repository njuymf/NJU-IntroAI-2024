import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 32 * 32),
            nn.Tanh()  # 输出在[-1, 1]之间
        )

    def forward(self, z):
        return self.model(z).reshape(-1, 3, 32, 32)  # 转换成3通道32x32图像


# 定义鉴别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出在[0, 1]之间，用于分类
        )

    def forward(self, img):
        return self.model(img)


# 超参数设置
learning_rate = 0.0002
batch_size = 64
num_epochs = 50

# 数据加载
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10已经是32x32，直接使用即可
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
])
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 创建生成器和鉴别器
generator = Generator()
discriminator = Discriminator()

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator.to(device)
discriminator.to(device)

# 损失函数与优化器
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# 训练
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)

        # 训练鉴别器
        optimizer_D.zero_grad()
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        outputs = discriminator(imgs)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(imgs.size(0), 100).to(device)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        optimizer_G.zero_grad()

        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)  # 生成器希望得到真实标签
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

    # 生成图像
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake_imgs = generator(torch.randn(16, 100).to(device)).cpu()
            plt.figure(figsize=(8, 4))
            for j in range(16):
                plt.subplot(4, 4, j + 1)
                plt.imshow((fake_imgs[j].permute(1, 2, 0) + 1) / 2)  # 将[-1, 1]转换到[0, 1]
                plt.axis('off')
            plt.show()
