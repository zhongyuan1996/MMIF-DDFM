from functools import partial
import os
import argparse
import yaml
import torch
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
from PIL import Image
from sewar.full_ref import vifp
from util.evaluators import calculate_entropy, calculate_standard_deviation, calculate_mi, calculate_ssim, reload_and_preprocess
warnings.filterwarnings('ignore')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str,default = 'configs/model_config_imagenet.yaml')
    parser.add_argument('--diffusion_config', type=str,default='configs/multi_diffusion_config.yaml')                     
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='/data/yuan/DDFM_dir/output')
    parser.add_argument('--input_dir', type=str, default='/data/yuan/DDFM_dir/input')
    parser.add_argument('--gls_config', type=str, default='configs/gls_config.yaml')
    #parser.add_argument('--U_ddpm_dir', type=str, defaule='/data/yuan/DDFM_dir/models/')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)  
    diffusion_config = load_yaml(args.diffusion_config)
    gls_config = load_yaml(args.gls_config)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # #Initialize layers for mg, ml, and ms
    # gls_model = create_mapping(**gls_config)
    # gls_model = gls_model.to(device)
    # gls_model.train()

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop_multimodal, model=model)
   
    # Working directory
    #test_folder=r"input"
    test_folder=args.input_dir     
    out_path = args.save_dir
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # optimizer = torch.optim.Adam(gls_model.parameters(), lr=1e-3)
    # criterion = torch.nn.MSELoss()

    i=0

    for img_name in os.listdir(os.path.join(test_folder,"ir")):
        inf_img = image_read(os.path.join(test_folder,"vi",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 
        vis_img = image_read(os.path.join(test_folder,"ir",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 

        inf_img = inf_img*2-1
        vis_img = vis_img*2-1
        # crop to make divisible
        scale = 32
        h, w = inf_img.shape[2:]

        h = h - h % scale
        w = w - w % scale
   
        inf_img = ((torch.FloatTensor(inf_img))[:,:,:h,:w]).to(device)
        vis_img = ((torch.FloatTensor(vis_img))[:,:,:h,:w]).to(device)
        assert inf_img.shape == vis_img.shape

        logger.info(f"Inference for image {i}")

        seed=1234
        torch.manual_seed(seed)
        x_start = torch.randn((inf_img.repeat(1, 3, 1, 1)).shape, device=device)

        #do mask prediction and loss function
        with torch.no_grad():
            sample_A = sample_fn(x_start=x_start, record=True, I = inf_img, V = vis_img, save_root=out_path, img_index = os.path.splitext(img_name)[0], lamb=0.5,rho=0.001)
            # sample_B = sample_fn(x_start=x_start, record=True, I = vis_img, V = inf_img, save_root=out_path, img_index = os.path.splitext(img_name)[0], lamb=0.5,rho=0.001)

        sample_A=sample_A.detach().cpu().squeeze().numpy()
        # sample_B=sample_B.detach().cpu().squeeze().numpy()
        sample_A = np.transpose(sample_A, (1,2,0))
        # sample_B = np.transpose(sample_B, (1,2,0))
        sample_A = cv2.cvtColor(sample_A,cv2.COLOR_RGB2YCrCb)[:,:,0]
        # sample_B = cv2.cvtColor(sample_B,cv2.COLOR_RGB2YCrCb)[:,:,0]
        sample_A = (sample_A - np.min(sample_A)) / (np.max(sample_A) - np.min(sample_A)) * 255
        # sample_B = (sample_B - np.min(sample_B)) / (np.max(sample_B) - np.min(sample_B)) * 255
        sample_A = sample_A.astype(np.uint8)
        # sample_B = sample_B.astype(np.uint8)
        image_A = Image.fromarray(sample_A)
        # image_B = Image.fromarray(sample_B)
        image_A.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]}_A.png"))
        # image_B.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]}_B.png"))
        i += 1

        # sample= sample.detach().cpu().squeeze().numpy()
        # sample=np.transpose(sample, (1,2,0))
        # sample=cv2.cvtColor(sample,cv2.COLOR_RGB2YCrCb)[:,:,0]
        # sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255
        # sample = sample.astype(np.uint8)
        # image = Image.fromarray(sample)
        # image.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]}.png"))
        # i += 1



    input_dir = args.input_dir
    output_dir = os.path.join(args.save_dir, 'recon')
    EN_list = []
    SD_list = []
    # Loop over images
    for img_name in os.listdir(os.path.join(input_dir, "ir")):
        path_ir = os.path.join(input_dir, "ir", img_name)
        path_vi = os.path.join(input_dir, "vi", img_name)
        path_fused = os.path.join(output_dir, os.path.splitext(img_name)[0] + "_A.png")

        # Reload and preprocess images
        inf_img = reload_and_preprocess(path_ir)
        vis_img = reload_and_preprocess(path_vi)
        fused_img = reload_and_preprocess(path_fused)

        # Calculate metrics
        EN = calculate_entropy(fused_img)
        SD = calculate_standard_deviation(fused_img)
        EN_list.append(EN)
        SD_list.append(SD)

    print(f"Average Entropy: {np.mean(EN_list)}")
    print(f"Average Standard Deviation: {np.mean(SD_list)}")
