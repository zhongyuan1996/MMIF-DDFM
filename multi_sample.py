from functools import partial
import os
import argparse
import yaml
import torch
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from guided_diffusion.ViT_util import create_mapping
from util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings
from PIL import Image
from sewar.full_ref import vifp
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
    parser.add_argument('--gpu', type=int, default=0)
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

    #Initialize layers for mg, ml, and ms
    gls_model = create_mapping(**gls_config)
    gls_model = gls_model.to(device)
    gls_model.train()

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

    optimizer = torch.optim.Adam(gls_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

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

        #calcualte mg, ms and ml for each modality
        i_g, i_l, i_s = gls_model(inf_img)
        v_g, v_l, v_s = gls_model(vis_img)

        #do mask prediction and loss function
        # with torch.no_grad():
        i_f = sample_fn(x_start=x_start, record=True, mg = i_g, ml = i_l, ms = v_s, save_root=out_path, img_index = os.path.splitext(img_name)[0], lamb=0.5, eta=0.001)
        v_f = sample_fn(x_start=x_start, record=True, mg = v_g, ml = v_l, ms = v_s, save_root=out_path, img_index = os.path.splitext(img_name)[0], lamb=0.5, eta=0.001)
        
        print("i_f requires grad:", i_f.requires_grad)
        print("v_f requires grad:", v_f.requires_grad)
        exit()

        loss_i = criterion(i_f, inf_img)
        loss_v = criterion(v_f, vis_img)
        loss = loss_i + loss_v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"Loss: {loss.item()}")

        #samve images of both modalities
        i_f = i_f.detach().cpu().squeeze().numpy()
        i_f = np.transpose(i_f, (1,2,0))
        i_f = cv2.cvtColor(i_f, cv2.COLOR_RGB2YCrCb)[:,:,0]
        i_f = (i_f - np.min(i_f)) / (np.max(i_f) - np.min(i_f)) * 255
        i_f = i_f.astype(np.uint8)
        i_image = Image.fromarray(i_f)
        i_image.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]+'if'}_i.png"))

        v_f = v_f.detach().cpu().squeeze().numpy()
        v_f = np.transpose(v_f, (1,2,0))
        v_f = cv2.cvtColor(v_f, cv2.COLOR_RGB2YCrCb)[:,:,0]
        v_f = (v_f - np.min(v_f)) / (np.max(v_f) - np.min(v_f)) * 255
        v_f = v_f.astype(np.uint8)
        v_image = Image.fromarray(v_f)
        v_image.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]+'if'}_v.png"))

        i += 1

        # sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)) * 255
        # sample = sample.astype(np.uint8)
        # image = Image.fromarray(sample)
        # image.save(os.path.join(out_path, 'recon', f"{os.path.splitext(img_name)[0]}.png"))
        # i += 1
