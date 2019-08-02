# ResNet-Video-Features
Temporal video features extracted from ImageNet pre-trained ResNet-152.

-------------------

## Extract Frames
Firstly, we need to extract the frames of each video by **FFmpeg**. 
- On Linux, please refer to the `extract_frames.sh` file.
- On Windows, the scripts are similar:
```ruby
for %f in (./Videos/*.mp4) do mkdir "./Frames_folders/%~nf"

for %f in (./Videos/*.mp4) do ffmpeg -i "./Videos/%f" "./Frames_folders/%~nf/%05d.jpg"
```

## Extract Featrues for Frames
Just run the `resnet_features.py` in Python:
```ruby
python resnet_features.py
```
