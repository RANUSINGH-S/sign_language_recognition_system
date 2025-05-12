# Sign Language Detection

A computer vision project that detects and recognizes sign language gestures using OpenCV.

## Features

- Real-time hand detection using skin color segmentation
- Data collection tool for creating custom sign language datasets
- Processing pipeline for standardizing hand images
- Simple and intuitive user interface

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install opencv-python numpy
   ```

## Usage

### Data Collection

Run the data collection script to capture images for training:

```
python datacollection.py
```

- Press 's' to save the current hand image
- Press 'q' to quit

### Testing

Run the test script to detect hand gestures in real-time:

```
python test.py
```

- Press 'q' to quit

## Project Structure

- `datacollection.py`: Script for collecting hand gesture images
- `test.py`: Script for real-time hand detection
- `Data/`: Directory containing collected hand gesture images

## License

MIT

## Acknowledgments

- OpenCV community
- Computer vision researchers
