from torch_fidelity import calculate_metrics
gen_dir = "./generated_images_cosine"
real_dir = "./processed_real_images"
metrics = calculate_metrics(
    input1=gen_dir,
    input2=real_dir,
    fid=True,  # 计算 FID
    isc=False  # 不计算 Inception Score
)
print("FID:", metrics["frechet_inception_distance"])