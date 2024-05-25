import os
import sys
import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from importlib import import_module

from torch.nn import functional as F
from core.utils import flow_viz
from core.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_exp_env():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True


def interp_imgs_for_video(ppl, origin_img0, origin_img1, saving_index):
    img0 = (torch.tensor(origin_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(origin_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    divisor = 2 ** (PYR_LEVEL-1+2)

    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)


    interp_img, bi_flow = ppl.inference(img0, img1,
            time_period=TIME_PERIOID,
            pyr_level=PYR_LEVEL)
    interp_img = interp_img[:, :, :h, :w]
    bi_flow = bi_flow[:, :, :h, :w]

    overlay_input = (ori_img0 * 0.5 + ori_img1 * 0.5).astype("uint8")
    interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    bi_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)

    flow01 = bi_flow[:, :, :2]
    flow10 = bi_flow[:, :, 2:]
    # flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
    # flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
    bi_flow = np.concatenate([flow01, flow10], axis=1)

    cv2.imwrite(os.path.join(SAVE_DIR, f'frame_{saving_index:04d}.jpg'), ori_img0)
    saving_index += 1
    cv2.imwrite(os.path.join(SAVE_DIR, f'frame_{saving_index:04d}.jpg'), interp_img)
    print("\nInterpolation is completed! Please see the results in %s" % (SAVE_DIR))
    saving_index += 1
    return saving_index

def load_images(directory, index):
    frame_number = f"{index:04d}"
    filename = f"frame_{frame_number}.jpg"  # Assuming the images are in .png format
    filepath = os.path.join(directory, filename)
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")

    parser.add_argument("--frame_path", type=str, default="/root/autodl-tmp/Outputs/dancing/dancing4x",                                                           
            help="file path of the images folder")
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period for interpolated frame")
    parser.add_argument("--save_dir", type=str,
            default="/root/autodl-tmp/Outputs/dancing/dancing8x",
            help="dir to save interpolated frame")

    ## load base version of UPR-Net by default
    parser.add_argument('--model_size', type=str, default="base",
            help='model size, one of (base, large, LARGE)')
    parser.add_argument('--model_file', type=str,
            default="./checkpoints/upr-base.pkl",
            help='weight of UPR-Net')

    # for inference only
    parser.add_argument("--number_of_frames", type=int, default=1000, help="number of frames in folder")



    args = parser.parse_args()
    TIME_PERIOID = args.time_period
    SAVE_DIR = args.save_dir
    DIRECTORY_PATH = args.frame_path
    number_of_images = args.number_of_frames

    #**********************************************************#
    # Start initialization
    init_exp_env()
    print("\nInitialization is OK! Begin to interp images...")
    #**********************************************************#
    # => parse args and init the training environment
    # global variable
    saving_index = 0
    for i in range(1, number_of_images):
        FRAME0 = load_images(directory=DIRECTORY_PATH, index=i)
        FRAME1 = load_images(directory=DIRECTORY_PATH, index=i+1)
        
        ori_img0 = cv2.imread(FRAME0)
        ori_img1 = cv2.imread(FRAME1)

    #**********************************************************#
    # => read input frames and calculate the number of pyramid levels
        test1 = ori_img0.shape 
        test2 = ori_img1.shape
        #    ValueError("Please ensure that the input frames have the same size!")

        width = ori_img0.shape[1]
        PYR_LEVEL = math.ceil(math.log2(width/448) + 3)

    #**********************************************************#
    # => init the pipeline and interpolate images
        model_cfg_dict = dict(
                load_pretrain = True,
                model_size = args.model_size,
                model_file = args.model_file
                )

        ppl = Pipeline(model_cfg_dict)
        ppl.eval()
        saving_index = interp_imgs_for_video(ppl, ori_img0, ori_img1, saving_index)
        print("saved once")

