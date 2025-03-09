## PigletCounting_STF_MLCM

This project is the code implementation for the paper titled **"Robust Piglet Counting in Crowded Environments via Small Trajectory Fusion and Multi-Line Counting Module."** The primary goal of this project is to develop an effective method for counting piglets in crowded environments using advanced computer vision techniques.
Here is the sample code.Additional features and improvements will be implemented following the publication of the associated research paper.

## Features

- **YOLOv8 Object Detection**: Utilizes the YOLOv8 model for real-time object detection, specifically tailored for piglet counting.
- **Trajectory Fusion**: Implements a small trajectory fusion technique to enhance the accuracy of counting in dense scenarios.
- **Multi-Line Counting Module**: Introduces a multi-line counting approach to improve the robustness of the counting process.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.9
- OpenCV
- NumPy
- Pandas
- PyCUDA
- TensorRT
- tqdm

You can install the required packages using pip:
'''bash
pip install -r requirements.txt
'''

## Usage
```bash
python3 v4.1_count.py demo/10146_20201218103004003_81.mp4 demo/output.mp4 . demo/output.csv 81
```
the output is like:
```base
[03/07/2025-11:44:47] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
write to demo/output.mp4 12 [1280, 720]
 99%|██████████████████████████████████████▌| 538/541.0 [00:15<00:00, 51.79it/s]count: -81
542it [00:16, 32.20it/s]  
```
you need to build engine file from onnx using YOLOv8-TensorRT:
```bash
cd YOLOv8-TensorRT
python3 build.py --weights ./models/voc_pig_v4.1/train/weights/best.onnx --iou-thres 0.7 --conf-thres 0.25 --topk 200 --device cuda:0
```

## Future Work

This project is currently a work in progress. Additional features and improvements will be implemented following the publication of the associated research paper. Stay tuned for updates!

## Acknowledgments

We would like to thank the contributors and researchers who have made this project possible. Your support and collaboration are greatly appreciated.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
