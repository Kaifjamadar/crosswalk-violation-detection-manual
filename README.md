# crosswalk-violation-detection-manual
🚸

crosswalk-violation-detection-manual
 is a computer vision system to monitor whether pedestrians are using zebra crossings or not while crossing roads. It supports both **manual** and **automatic** zone detection using YOLO models.

## 🔧 Modes

- **manual_mode.py**: User manually defines the zebra crossing zone.

## 📁 Project Structure

- `models/` – YOLO models for pedestrian and zebra crossing detection
- `source video` – video for detection of people using crosswalk or not
- `main.py` – Optional script to manually choose edges of crossing 

## 🚀 How to Run

Install dependencies:
```bash
pip install -r requirements.txt
