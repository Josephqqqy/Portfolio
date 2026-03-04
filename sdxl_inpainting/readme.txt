python train_image_inpainting_lora_sdxl2.py \
    --pretrained_model_name_or_path="sdxl-inpainting" \
    --dataset_name="sdxl-inpainting-lighting" \
    --resolution=768 \
    --train_batch_size=1 \
    --num_train_epochs=6 \
    --learning_rate=1e-4 \
    --output_dir="./lora_sdxl_inpainting_linear2"\
    --lr_scheduler="linear"
    --lr_warmup_steps=500

    --rank=8 \
    --lora_alpha=8 \    已完成
    --lr_scheduler="cosine" \ "constant_with_warmup"\"linear" 已完成linear 1e-4衰减
    --lr_warmup_steps = 500 \ 300 600 800
    --snr_gamma = 5.0 \
    --learning_rate=1e-5 \
    --adam_weight_decay=0.015 \  如果希望训练更加稳定且模型不容易过拟合（加强正则化）

pip uninstall -y diffusers
pip install git+https://github.com/huggingface/diffusers.git

训练超参数

--seed
说明： 随机种子，用于结果可重复性。
默认值： None

--train_batch_size
说明： 每个设备上的训练批量大小。
默认值： 16
实验调节： 调整批量大小可以影响模型收敛和显存占用。

--num_train_epochs
说明： 总训练周期数。
默认值： 100
实验调节： 根据任务需求修改训练周期长短。

--max_train_steps
说明： 总训练步数（如果设置，则覆盖 epoch 数）。
默认值： None

--gradient_accumulation_steps
说明： 累积梯度的步数，达到一定步数后再进行参数更新。
默认值： 1
实验调节： 提高该值可以在显存受限时增大有效批量，但可能影响收敛速度。

--gradient_checkpointing
说明： 是否启用梯度检查点以节省显存。
默认值： False

--learning_rate
说明： 初始学习率。
默认值： 1e-4
实验调节： 学习率是影响模型收敛的重要因素，不同任务和批量大小下可能需要不同的值。

--rank
指的是低秩分解中低秩矩阵的维度，也可以理解为“低秩更新”的容量大小。
rank 越大，低秩矩阵的容量就越高，模型可以通过 LoRA 学到更多的新任务特定信息，但同时也会增加训练参数量和过拟合风险。
rank 越小，更新容量更有限，模型更新会更温和，更好地保留预训练知识，但可能在新任务上适应能力不足。
默认：4

--lora_alpha
lora_alpha 是 LoRA（低秩适应）中的一个超参数，它用于缩放低秩更新矩阵的贡献
较大值：如果 lora_alpha 较大，低秩更新部分对模型输出的影响就会更强。这样在微调时，模型会更大幅度地偏离预训练权重，以适应新任务。
较小值：较小的 lora_alpha 会让更新更温和，保持预训练模型的大部分知识，从而降低因过度更新而导致的训练不稳定风险，特别是在数据较少时。
--scale_lr
说明： 是否根据 GPU 数量、累积步数和批量大小自动缩放学习率。
默认值： False

--lr_scheduler
说明： 学习率调度器类型。可选项包括 "linear"（线性衰减直至训练结束时降低到预设的最低值（通常为零）。这种方式简单直观，适用于训练过程中希望稳定地降低学习率的场景）, 
"cosine"(余弦衰减从初始值下降到接近零。这种方式能提供平滑的衰减曲线，有助于在训练后期保持较低的更新幅度，通常有助于模型收敛到更好的局部最优), 
"cosine_with_restarts"（基于余弦衰减，但在一定周期后会“重启”，即重新回到初始的较高学习率，然后再开始新的余弦衰减周期。这样可以在训练中期偶尔“跳出”局部最优，增加探索的可能性）
"polynomial", 
"constant",
"constant_with_warmup"（先进行“预热”阶段，在这段时间内学习率从零（或一个非常低的值）逐渐线性上升到设定的常数值，之后保持恒定。warmup阶段有助于在训练初期稳定参数更新，减少大梯度带来的不稳定风险）。
默认值： "constant"(不变)      
实验调节： 调整调度器类型和预热步数可以获得不同的收敛曲线。

--lr_warmup_steps
说明： 学习率预热步数。
默认值： 500

--snr_gamma
说明： 用于损失重平衡的 SNR 权重 gamma，用来控制对各个时间步损失的重新加权，帮助模型更均衡地学习去噪和图像修复任务
在你的脚本中，如果设置了 snr_gamma，代码会计算每个时间步的 SNR，然后根据 SNR 和 snr_gamma 生成一个权重，用于调节均方误差（MSE）损失。这有助于在整个扩散过程上平衡损失，提升训练稳定性和修复效果
参见论文 https://arxiv.org/abs/2303.09556。
默认值： None
实验调节： 适当设置这个参数可能帮助模型更好地处理噪声预测问题。

--allow_tf32
说明： 是否允许在 Ampere GPU 上使用 TF32 加速运算。
默认值： False

--dataloader_num_workers
说明： 数据加载时使用的子进程数。
默认值： 0

--use_8bit_adam
说明： 是否使用 8-bit Adam 优化器以降低显存占用。
默认值： False

--adam_beta1、--adam_beta2
说明： Adam 优化器的 beta 参数。
默认值： 0.9 和 0.999

--adam_weight_decay
说明： 权重衰减系数。
正则化效果
通过在每次参数更新时对权重施加一个额外的惩罚项（通常是权重的 L2 范数），使权重值不会变得过大，从而有助于防止过拟合。
稳定训练
较小的权重可以使得模型在训练过程中更加稳定，避免梯度过大导致的震荡。
控制模型复杂度
权重衰减鼓励模型保持较小的权重值，这有助于降低模型的复杂度，使其泛化能力更强。
默认值： 1e-2

--adam_epsilon
说明： Adam 优化器中的 epsilon 值。
默认值： 1e-08

--max_grad_norm
说明： 梯度裁剪的最大范数。
默认值： 1.0




图像预处理和增强


--resolution
说明： 输入图像的分辨率，训练前所有图像都会被调整到该尺寸。
默认值： 1024
实验调节： 降低或提高分辨率会直接影响生成图像的细节和训练速度。

--center_crop
说明： 是否采用中心裁剪（开启则采用中心裁剪，否则随机裁剪）。
默认值： False
实验调节： 可以比较中心裁剪与随机裁剪对模型生成结果的影响。

--random_flip
说明： 是否对图像进行随机水平翻转。
默认值： False
实验调节： 开启数据增强可能提高模型的鲁棒性。


验证与输出相关参数

--validation_prompt
说明： 用于验证生成图像的提示文本。
默认值： None

--num_validation_images
说明： 每次验证生成的图像数量。
默认值： 4

--validation_epochs
说明： 每隔多少个 epoch 进行一次验证。
默认值： 1

--output_dir
说明： 保存模型、检查点及日志的输出目录。
默认值： "sd-model-finetuned-lora"

--cache_dir
说明： 模型和数据集下载的缓存目录。
默认值： None


模型与数据相关参数

--pretrained_model_name_or_path
说明： 必填，预训练模型的路径或 Hugging Face 上的标识。
默认值： 无（必须指定）

--pretrained_vae_model_name_or_path
说明： 用于数值稳定性的预训练 VAE 模型。
默认值： None

--revision、--variant
说明： 用于指定模型版本或变体（例如 fp16）。
默认值： None

--dataset_name
说明： 数据集名称，可以使用 Hugging Face Hub 上的公开数据集或自定义数据集。
默认值： None（需指定，否则应提供 --train_data_dir）

--dataset_config_name
说明： 数据集配置。
默认值： None

--train_data_dir
说明： 本地训练数据文件夹（如果不使用 dataset_name）。
默认值： None

--image_column
说明： 数据集中存放图像的列名。
默认值： "image"

--caption_column
说明： 数据集中存放文本描述的列名。
默认值： "text"






