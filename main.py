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
import time


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description='VadCLIP Abnormal Detection')
    parser.add_argument('--seed', default=234, type=int)
    parser.add_argument('--det_model_path', default='pretrained/best_noclip.pt')
    parser.add_argument('--classify_model_path', default='pretrained/best_noclip.pt')
    parser.add_argument('--det_fps', default=2, type=int)
    parser.add_argument('--classify_fps', default=2, type=int)

    parser.add_argument('--out_dir', default='exp_output')
    parser.add_argument('--camera', default="videos/video_2fps.mp4", help='Source of camera or video file path.')
    parser.add_argument('--subsample', default=False, action="store_true", help='Ignore frame to match target FPS. Only work if input is video')
    parser.add_argument('--visualize', default=False, action="store_true", help='Ignore frame to match target FPS. Only work if input is video')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ######## load model ########
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_model = load_model(args.det_model_path, device)
    det_model.load_clip_model("ViT-B/16")
    det_model.eval()
    ############################

    ######## load model ########
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classify_model = load_model(args.classify_model_path, device)
    classify_model.load_clip_model("ViT-B/16")
    classify_model.eval()
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
        if args.det_fps != original_fps:
            if not args.subsample:
                raise Exception(f"Input video is {original_fps} while the detection fps is {args.det_fps}. Considering using subsample")
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
        det_frame_interval = int(round(original_fps / args.det_fps))
    else:
        det_frame_interval = 1


    if args.classify_fps != original_fps:
        if not args.subsample:
            raise Exception(f"Input video is {original_fps} while the classification fps is {args.classify_fps}. Considering using subsample")
        else:
            # If input video != target FPS, use subsample
            use_subsample = True
    else:
        use_subsample = False


    if use_subsample:
        classify_frame_interval = int(round(original_fps / args.classify_fps))
    else:
        classify_frame_interval = 1
    ##############################
    
    ##############################


    out_dir = f'{args.out_dir}'
    os.makedirs(out_dir, exist_ok=True)

    det_local_feat_queue = deque(maxlen=det_model.attn_window)  # local window queue
    det_global_feat_queue = deque(maxlen=det_model.visual_length)  # global window queue

    classify_local_feat_queue = deque(maxlen=classify_model.attn_window)  # local window queue
    classify_global_feat_queue = deque(maxlen=64)

    gif_det_frames = []
    gif_classify_frames = []


    is_detect = False
    det_prob_queue = deque(maxlen=10)  # local window queue
    
    with torch.inference_mode():
        frame_idx = 0

        while True:
            ret, frame = cam.read()

            if not ret :
                break

            if is_detect:
                frame_interval = classify_frame_interval
            else:
                frame_interval = det_frame_interval
            
            if not is_detect:
                if frame_idx % frame_interval == 0: 
                    print(f"Detection at {frame_idx}")

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    feature = det_model.compute_clip_emb(frame) # including preprocess image and calculate clip feature
                    
                    det_local_feat_queue.append(feature) # push to local window

                    if len(det_local_feat_queue) < det_model.attn_window:
                        # wait until the window full
                        frame_idx += 1
                        continue

                    visual_features = torch.stack(list(det_local_feat_queue), dim=1)
                    local_features = det_model.encode_local_window(visual_features)

                    det_global_feat_queue.append(local_features[-1:])  #
                    
                    global_features = torch.cat(list(det_global_feat_queue), dim=1)
                    logit_features, _ = det_model.encode_global_window(global_features, False)

                    # # Get latest predictions
                    prob1 = torch.sigmoid(logit_features.flatten())

                    det_prob_queue.append(prob1[-1:])
                    prob_tensor = torch.stack(list(det_prob_queue), dim=1)  # shape [1, T]

                    # Condition 1: all recent probs > threshold
                    above_thresh = ((torch.all(prob_tensor > 0.9)) or prob1[-1].item() > 0.99) and len(det_prob_queue) > 5
                    # Condition 2: detect sharp increase (e.g., last prob - second last)
                    if prob_tensor.shape[1] >= 2:
                        prob_diff = prob_tensor[0, -1] - prob_tensor[0, -2]  # scalar
                    else:
                        prob_diff = torch.tensor(0.0)

                    sharp_increase = prob_diff > 0.5  # Tune this threshold

                    if above_thresh or sharp_increase:
                        is_detect = True

                    if args.visualize:
                        prob_text = f"{prob1[-1:].item():.4f}"
                        cv2.putText(frame, prob_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 1)

                        gif_det_frames.append(frame)
            else:
                if frame_idx % frame_interval == 0: 
                    print(f"Classification at {frame_idx}")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    feature = classify_model.compute_clip_emb(frame) # including preprocess image and calculate clip feature

                    classify_local_feat_queue.append(feature) # push to local window

                    if len(classify_local_feat_queue) < classify_model.attn_window:
                        # wait until the window full
                        frame_idx += 1
                        continue

                    visual_features = torch.stack(list(classify_local_feat_queue), dim=1)
                    local_features = classify_model.encode_local_window(visual_features)

                    classify_global_feat_queue.append(local_features[-1:])  #
                    
                    global_features = torch.cat(list(classify_global_feat_queue), dim=1)
                    logit_features, logits2 = classify_model.encode_global_window(global_features, True)


                    probs = F.softmax(logits2[:, -1:].flatten(0, 1), dim=1)  # shape [B, T, C]
                    predicted_indices = torch.argmax(probs, dim=1)  # shape [B, T]

                    # Expand predicted_indices to match probs for gather
                    predicted_probs = torch.gather(probs, 1, predicted_indices.unsqueeze(-1)).squeeze(-1)  # shape [B, T]
                    predicted_class_names = [classify_model.prompt_text[int(i)] for i in predicted_indices.cpu()]

                    if args.visualize:
                        prob_text = f"{predicted_probs.item():.4f}"
                        cv2.putText(frame, f"{predicted_class_names[0]} - {prob_text}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 0, 0), 1)

                        gif_classify_frames.append(frame)

                    if predicted_class_names[0] == "Normal":
                        is_detect = False

            frame_idx += 1

    cam.release()

    if args.visualize:
        output_gif_path = f'{out_dir}/{os.path.basename(args.camera) + ".gif"}'


        # Convert frames to PIL Images
        gif_det_frames_pil = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in gif_det_frames]
        gif_classify_frames_pil = [Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in gif_classify_frames]

        # Calculate durations (ms)
        det_duration = int(1000 / args.det_fps)        # 2 fps
        classify_duration = int(1000 / args.classify_fps)  # 30 fps

        # Concatenate frames and durations
        all_frames = gif_det_frames_pil + gif_classify_frames_pil
        durations = [det_duration] * len(gif_det_frames_pil) + [classify_duration] * len(gif_classify_frames_pil)

        # Save using PIL
        all_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=durations,
            loop=0
        )