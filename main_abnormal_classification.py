import argparse
import cv2
import os
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from PIL import Image
import imageio

from model.model import load_model
from collections import deque



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='VadCLIP Abnormal Classification')
    parser.add_argument('--seed', default=234, type=int)
    parser.add_argument('--model_path', default='pretrained/best.pt')
    parser.add_argument('--out_dir', default='exp_output')
    parser.add_argument('--target_fps', default=15, type=int)
    parser.add_argument('--camera', default="videos/videp_30fps.mp4", help='Source of camera or video file path.')
    parser.add_argument('--subsample', default=False, action="store_true", help='Ignore frame to match target FPS. Only work if input is video')
    parser.add_argument('--visualize', default=False, action="store_true", help='Ignore frame to match target FPS. Only work if input is video')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ######## load model ########
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)
    model.load_clip_model("ViT-B/16")
    model.eval()
    ############################
    
    ######## CAMERA STUFF ########
    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        # cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
        cam = cv2.VideoCapture(cam_source)
        if not cam.isOpened():
            raise RuntimeError(f"Cannot open camera/video source: {cam_source}")

        original_fps = cam.get(cv2.CAP_PROP_FPS)
        if args.target_fps != original_fps:
            if not args.subsample:
                raise Exception(f"Input video is {original_fps} while the target fps is {args.target_fps}. Considering using subsample")
            else:
                # If input video != target FPS, use subsample
                use_subsample = True
        else:
            use_subsample = False

    else:
        # Use normal thread loader for webcam.
        raise Exception("still not implemented")
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    if use_subsample:
        frame_interval = int(round(original_fps / args.target_fps))
        print(f"Using frame_interval = {frame_interval}")
    else:
        frame_interval = 1
    ##############################


    out_dir = f'{args.out_dir}'
    os.makedirs(out_dir, exist_ok=True)


    local_feat_queue = deque(maxlen=model.attn_window)  # local window queue
    global_feat_queue = deque(maxlen=model.visual_length)  # global window queue
    gif_frames = []
      
    with torch.inference_mode():
        frame_idx = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0: 
                print(f"Classification at {frame_idx}")
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                feature = model.compute_clip_emb(frame) # including preprocess image and calculate clip feature
                local_feat_queue.append(feature) # push to local window

                if len(local_feat_queue) < model.attn_window:
                    # wait until the window full
                    frame_idx += 1
                    continue

                visual_features = torch.stack(list(local_feat_queue), dim=1)
                local_features = model.encode_local_window(visual_features)

                global_feat_queue.append(local_features[-1:])  #
                
                global_features = torch.cat(list(global_feat_queue), dim=1)
                logit_features, logits2 = model.encode_global_window(global_features, True)

                # # Get latest predictions
                prob1 = torch.sigmoid(logit_features[:, -1:].flatten())

                probs = F.softmax(logits2[:, -1:].flatten(0, 1), dim=1)  # shape [B, T, C]
                predicted_indices = torch.argmax(probs, dim=1)  # shape [B, T]

                # Expand predicted_indices to match probs for gather
                predicted_probs = torch.gather(probs, 1, predicted_indices.unsqueeze(-1)).squeeze(-1)  # shape [B, T]
                predicted_class_names = [model.prompt_text[int(i)] for i in predicted_indices.cpu()]

                if args.visualize:
                    prob_text = f"{prob1.item():.4f}"
                    cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 1)

                    prob_text = f"{predicted_probs.item():.4f}"
                    cv2.putText(frame, f"{predicted_class_names[0]} - {prob_text}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 0, 0), 1)

                    gif_frames.append(frame)

            frame_idx += 1
            

    cam.release()

    if args.visualize:
        output_gif_path = f'{out_dir}/{os.path.basename(args.camera) + ".gif"}'
        imageio.mimsave(output_gif_path, gif_frames, fps=args.target_fps)