# Cycling Overlay Generator

Create professional-looking cycling data overlays for your ride videos using FIT file data. This tool generates a green screen overlay with real-time metrics including power, speed, cadence, heart rate, and GPS tracking.

## Features

- **Power**: Vibrant gradient power display with smooth animations
- **Speed Gauge**: Circular gauge showing current speed in km/h
- **Cadence Gauge**: RPM display with color-coded intensity
- **Heart Rate Monitor**: BPM display with dynamic color coding
- **GPS Map**: Live route tracking with trail visualization
- **Green Screen Background**: Easy to overlay on your ride footage
- **Smooth Animations**: All metrics use easing for professional-looking transitions

## Example Output

The overlay displays:
- GPS route map (top left)
- Heart rate (top center)
- Power bar with gradient (bottom center)
- Speed gauge (bottom left)
- Cadence gauge (bottom right)

All on a green screen background for easy chroma keying into your videos.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7 or higher**
- **pip** (Python package installer)

## Installation

###  1. Install Required Python Packages

```bash
pip install fitparse pillow opencv-python numpy
```

**Package descriptions:**
- `fitparse`: Parse FIT files from cycling computers
- `pillow`: Image processing and drawing
- `opencv-python`: Video generation
- `numpy`: Numerical operations

### 2. Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import fitparse, PIL, cv2, numpy; print('All packages installed successfully!')"
```

## Usage

### 1. Prepare Your FIT File

Place FIT file from your race in the project directory and name it `ride.fit`

### 2. Configure the Script

Open `make_overlay.py` and modify the configuration section at the top:

```python
# ---------- CONFIG ----------
FIT_FILENAME = "ride.fit"          # Your FIT file name
OUTPUT_VIDEO = "overlay_greenscreen.mp4"  # Output video name
WIDTH, HEIGHT = 1920, 1080         # Video resolution
FPS = 30                            # Frames per second
```

### 3. Generate the Overlay

Run the script:

```bash
python make_overlay.py
```
*** NOTE: The longer the video, the longer this process takes. A ~45 minute video at 30 frames per second requires ~81,000 frames to be rendered. ***

You'll see progress output like:

```
Loading FIT file: ride.fit
Ride: 45.2 min | Frames: 81,360 @ 30 fps
Pre-processing...
Initializing video writer...

Rendering overlay...

  12.5% | Frame 10,170/81,360 | 25.3 fps | ETA: 2.8 min
  ...
```

### 4. Find Your Output

The generated video will be saved as `overlay_greenscreen.mp4` in the same directory.

## Customization

### Color Schemes

Edit the color definitions in `make_overlay.py`:

```python
# Power gradient colors
POWER_COLOR_LOW = (100, 200, 255)   # Bright blue
POWER_COLOR_MID = (150, 100, 255)   # Purple
POWER_COLOR_HIGH = (255, 50, 150)   # Pink

# Speed colors
SPEED_COLOR_LOW = (0, 255, 200)     # Cyan
SPEED_COLOR_HIGH = (100, 150, 255)  # Blue

# Heart rate colors
HR_COLOR_LOW = (255, 200, 0)        # Yellow
HR_COLOR_HIGH = (255, 50, 50)       # Red

# Cadence colors
CADENCE_COLOR_LOW = (255, 100, 200)  # Pink
CADENCE_COLOR_HIGH = (200, 50, 255)  # Purple
```

### Animation Speed

Adjust the easing strength for faster/slower animations:

```python
EASING_STRENGTH = 0.2  # Lower = smoother but slower, Higher = snappier
```

### Video Settings

```python
WIDTH, HEIGHT = 1920, 1080  # Change to match your video resolution
FPS = 30                     # 30 fps for faster rendering, 60 for smoother
```

### Font Size

Modify font sizes in the `fonts` dictionary (around line 461):

```python
fonts = {
    "huge": get_font(70, bold=True),    # Power value
    "big": get_font(56, bold=True),     # Speed/Cadence values
    "medium": get_font(38, bold=True),  # Heart rate
    "small": get_font(24),              # Units (km/h, rpm)
    "tiny": get_font(18),               # Units (bpm)
}
```

## Troubleshooting

### Video is Too Long to Process

Reduce the frame rate for faster rendering:
```python
FPS = 24  # Instead of 30 or 60
```

### Memory Issues with Long Rides

The script processes the entire ride. For very long rides (3+ hours), consider:
- Reducing FPS
- Reducing resolution
- Processing the ride in segments

## Performance Tips

- **Lower FPS**: Use 24-30 fps instead of 60 for faster rendering
- **Lower Resolution**: Use 1280x720 if you don't need 1080p
- **Close Other Apps**: Video rendering is CPU-intensive
- **Expected Time**: A 1-hour ride at 30fps takes roughly 5-10 minutes to render on a modern computer

## File Structure

```
cyclingoverlay/
├── make_overlay.py      # Main script
├── ride.fit                    # Your FIT file (add this)
├── overlay_greenscreen.mp4    # Generated output
└── README.md                   # This file
```

## Technical Details

### FIT File Data Used

- **Power**: Watts from power meter
- **Speed**: Converted from m/s to km/h
- **Cadence**: RPM from cadence sensor
- **Heart Rate**: BPM from heart rate monitor
- **GPS**: Latitude/longitude for route mapping
- **Distance**: Total distance traveled

### Video Specifications

- **Resolution**: 1920x1080 (configurable)
- **Frame Rate**: 30 fps (configurable)
- **Background**: RGB (0, 255, 0) - Pure green for chroma keying
- **Codec**: MP4V (compatible with most video editors)

### Data Interpolation

The script interpolates between FIT file data points (typically recorded every second) to create smooth animations at the specified frame rate. All metrics use easing functions for professional-looking transitions.

## Contributing

Feel free to fork this repository and submit pull requests with improvements!

## License

This project is open source and available under the MIT License.
