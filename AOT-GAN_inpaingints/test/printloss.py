# 用于输出训练过程中的的损失变化
import json
import matplotlib.pyplot as plt

def printloss():
    # 存储每个周期的损失
    epoch_losses = []
    num_epochs = 135
    # 读取每个周期的平均损失数据
    for epoch in range(40,num_epochs):
        with open(f'./model_losses/average_losses_epoch_{epoch}.json', 'r') as f:
            avg_losses = json.load(f)
            epoch_losses.append(avg_losses)

    # 提取各损失类型的数据
    L1_losses = [epoch['L1'] for epoch in epoch_losses]
    Style_losses = [epoch['Style'] for epoch in epoch_losses]
    Perceptual_losses = [epoch['Perceptual'] for epoch in epoch_losses]
    advg_losses = [epoch['advg'] for epoch in epoch_losses]
    advd_losses = [epoch['advd'] for epoch in epoch_losses]

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(40, num_epochs), L1_losses, label='L1 Loss')
    plt.plot(range(40, num_epochs), Style_losses, label='Style Loss')
    plt.plot(range(40, num_epochs), Perceptual_losses, label='Perceptual Loss')
    plt.plot(range(40, num_epochs), advg_losses, label='Generator Adversarial Loss')
    plt.plot(range(40, num_epochs), advd_losses, label='Discriminator Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()