# MaiZaic: End-to-End Pipeline for Mosaicking UAV-Captured Aerial Imagery

MaiZaic is an advanced mosaicking algorithm designed to process freely flown aerial imagery captured by UAVs. 

## 5 Novel Contributions

1. **Dynamic Sampling**
   - Optimizes frame selection based on UAV movements using optical flow.
   - Sampling based on overlap for forward, backward, and sideways, and more frequency on more complicated movements such as rotation and zoom.

2. **Automatic Lens Calibration**
   - automatic calibration of the camera lens and accounts for gimbal misalignment due to wear and tear.
   - For a more accurate image alignment and reduce distortion in the final mosaic.

3. **Homography Estimation with Cornetv3**
   - Utilizes the Cornetv3 architecture for a more robust and faster homography estimation.
   - Improves accuracy by trained the new model with a more balanced movements, different intensities, augmented with synthetic color changes, and different maize growth and soil colors.

4. **Shot Detection via UAV Movement Analysis**
   - Detects shot boundaries by monitoring changes in UAV movement.
   - Grouping the same UAV movements in the same mini mosaics.

5. **Mini Mosaicking to Minimize Error Accumulation**
   - Processes images in smaller sections (mini mosaics) to limit error propagation.
   - Higher accuracy in large-scale mosaics.

## Getting Started

### Prerequisites
- Python 2.x
- OpenCV
- Tensorflow
- NumPy
- Other dependencies as specified in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dek8v5/MaiZaic
   ```
2. Navigate to the project directory:
   ```bash
   cd MaiZaic/code
   ```
3. Install the required packages
   ```bash
   sudo install -r requirement.txt
   ```

### Running MaiZaic
1. locate your video that you want to mosaic

2. dynamic sample first
   ```bash
   python dynamic_sampling.py -video /path/to/raw/video -save_path /path/to/where/you/want/to/save/the/quiver_and_raw_frames/ -scale int -fps int -fname str 
   ```
3. calibrate the frame extracted from the dynamic sampling
   ```bash
   python calibration.py -image_path /path/to/your/raw/frames -save_path /path/to/the/working/directory
   ```
4. run the calibrated frames to the homography estimator of your choice
   ```bash
   example using surf:
   
   cd surf
   python surf_homography_estimation.py -image_path /path/to/your/calibrated/frames -save_path /path/to/the/working/directory -scale int (for resizing the frame dimension)
   
   ```
5. before running the mini mosaicking script, change the working directory on top of maizaic_run.sh
    
6. run the mini mosaic script
   ```bash
   ./maizaic_run.sh
   ```
