
# FOR DETECTION, USE MODEL with 2FPS
python main_abnormal_detection.py --visualize --camera videos/video_2fps.mp4 \
                                       --model_path pretrained/best_noclip.pt \
                                       --subsample --target_fps 2 --out_dir exp_output

# FOR CLASSIFICATION, USE MODEL with 15 FPS, subsample from 30FPS video

python main_abnormal_classification.py --visualize --camera videos/video_30fps.mp4 \
                                       --model_path pretrained/best_auc_no_clip.pt \
                                       --subsample --target_fps 15 --out_dir exp_output
