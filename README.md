# Signature-Verification-System

A desktop-based signature verification application that uses **ORB feature extraction** and **Lowe's Ratio Test** to compare two signature images and determine if they match. Built using Python, OpenCV, and a sleek GUI powered by Tkinter and `ttkbootstrap`.

## Features

- ORB (Oriented FAST and Rotated BRIEF) feature detection
- Lowe’s Ratio Test for accurate matching
- Real-time image preview and preprocessing (grayscale + Gaussian blur)
- Animated progress bar showing match confidence
- Save visual match results to file
- Adjustable match threshold
- GUI powered by Tkinter + ttkbootstrap (Dark Theme)

---

## Tech Stack

- **Language:** Python 3.x  
- **Libraries:** OpenCV, NumPy, Tkinter, ttkbootstrap, PIL (Pillow)

---

## How It Works

1. **Preprocessing:**
   - Converts input images to grayscale
   - Applies Gaussian Blur to reduce noise

2. **Feature Extraction:**
   - ORB detects keypoints and computes descriptors

3. **Matching:**
   - Brute-Force Matcher (BFMatcher) with Hamming distance
   - Uses Lowe’s Ratio Test to filter good matches

4. **Result:**
   - Draws top 20 matches
   - Displays a match/mismatch result based on a user-defined threshold

---

## Installation

### Prerequisites

- Python 3.x
- pip

### Install Dependencies

```bash
pip install opencv-python numpy ttkbootstrap pillow

