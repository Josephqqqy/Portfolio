## 用于展示图像修复效果
import matplotlib.pyplot as plt
import torch
from model.model import InpaintGenerator
from  Dataload.dataload import InpaintingData
from  test.tensor2im import tensor2im
from attrdict import AttrDict
def showimg():
    args = {
        "dir_image": "./datasets/test/face",
        "data_train": "celeb",
        "dir_mask": "./datasets/test/face-mask",
        "mask_type": "pconv",
        "image_size": 512,
    }
    args = AttrDict(args)
    fig,axes = plt.subplots(1,4,figsize=(20,20))
    model = InpaintGenerator()
    model.eval()
    model.load_state_dict(torch.load('./training_checkpoints/checkpoint_epoch_135.pth')['generator_state_dict'])
    data = InpaintingData(args)
    img, mask,filename = data[0]
    masked_image = (img * (1 - mask).float()) + mask
    masked_img = torch.cat((masked_image, mask), dim=0)
    input = masked_img.unsqueeze(0)
    output = model(input).detach()
    original = tensor2im(img.unsqueeze(0))
    mask = tensor2im(mask.unsqueeze(0))
    masked_original = tensor2im(masked_image.unsqueeze(0))
    inpainting = tensor2im(output)
    print(filename)
    plt.figure()
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[1].imshow(masked_original)
    axes[1].set_title('Masked Original')
    axes[2].imshow(inpainting)
    axes[2].set_title('Inpainting')
    axes[3].imshow(mask)
    axes[3].set_title('mask')
    plt.show()