
# # FOR DETECTION, USE MODEL with 2FPS
# python main_abnormal_detection.py --visualize --camera /workspace/silverai/abnormal_det/code/video/accident_2.MP4 \
#                                        --model_path pretrained/best_noclip.pt \
#                                        --subsample --target_fps 2 --out_dir exp_output

# FOR CLASSIFICATION, USE MODEL with 15 FPS, subsample from 30FPS video

# python main_abnormal_classification.py --visualize --camera /workspace/silverai/abnormal_det/code/video/explosion.MP4 \
#                                        --model_path pretrained/best_auc_no_clip.pt \
#                                        --subsample --target_fps 15 --out_dir exp_output


python main.py --visualize --camera videos/video_30fps.mp4 \
                --det_model_path pretrained/best_noclip.pt --det_fps 2 \
                --classify_model_path pretrained/best_auc_no_clip.pt --classify_fps 15 \
                --subsample  --out_dir exp_output