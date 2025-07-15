import cv2 
import numpy as np
import time
import os
import tkinter as tk
from PIL import Image, ImageTk
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox

# ---------- Image Preprocessing ----------
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

# ---------- ORB Feature Extraction ----------
def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# ---------- Match using Lowe's Ratio Test ----------
def match_signatures(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches, matches

# ---------- Save Match Result Image ----------
def save_result_image(image):
    output_path = "match_result.png"
    cv2.imwrite(output_path, image)
    log(f"Match image saved to {output_path}")

# ---------- GUI Callback to Verify ----------
def verify_and_display():
    log_box.delete(1.0, tk.END)
    result_label.config(text="")
    progress_bar['value'] = 0
    progress_bar.update_idletasks()

    if not reference_path.get() or not test_path.get():
        messagebox.showwarning("Input Missing", "Please select both signature images.")
        return

    try:
        log("Loading and processing images...")
        ref_img = preprocess_image(reference_path.get())
        test_img = preprocess_image(test_path.get())

        kp1, des1 = extract_features(ref_img)
        kp2, des2 = extract_features(test_img)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            messagebox.showerror("Error", "Insufficient features in one or both images.")
            return

        good_matches, matches = match_signatures(des1, des2)
        threshold = match_threshold.get()
        match_ratio = len(good_matches) / threshold if threshold > 0 else 0
        percentage = int(min(100, match_ratio * 100))

        style_name = "success" if len(good_matches) >= threshold else "danger"
        progress_bar.configure(bootstyle=style_name)

        for val in range(0, percentage + 1, 2):
            progress_bar['value'] = val
            root.update_idletasks()
            time.sleep(0.01)

        result_text = "MATCH ✅" if len(good_matches) >= threshold else "MISMATCH ❌"
        result_label.config(text=f"{result_text} ({len(good_matches)} good matches)",
                            foreground="green" if len(good_matches) >= threshold else "red")
        log(f"Result: {result_text} - {len(good_matches)} good matches")

        matched_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, good_matches[:20], None, flags=2)

        max_width = 1000
        scale_ratio = max_width / matched_img.shape[1]
        resized_img = cv2.resize(matched_img, None, fx=scale_ratio, fy=scale_ratio)

        log("Displaying visual match result...")
        cv2.imshow("Visual Match Result", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        save_result_image(resized_img)

    except Exception as e:
        messagebox.showerror("Error", str(e))
        log(f"Error: {str(e)}")
    finally:
        progress_bar.update_idletasks()

# ---------- Browse Functions ----------
def browse_reference():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    reference_path.set(path)
    preview_image(path, reference_canvas)

def browse_test():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    test_path.set(path)
    preview_image(path, test_canvas)

# ---------- Preview Image on Canvas ----------
def preview_image(path, canvas):
    try:
        img = Image.open(path).resize((160, 100))
        img = ImageTk.PhotoImage(img)
        canvas.image = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
    except:
        canvas.delete("all")

# ---------- Log Messages ----------
def log(message):
    log_box.insert(tk.END, message + "\n")
    log_box.see(tk.END)

# ---------- GUI Setup ----------
root = tb.Window(themename="darkly")
root.title("✍️ Signature Verification System - ORB")
root.geometry("900x700")

reference_path = tk.StringVar()
test_path = tk.StringVar()
match_threshold = tk.IntVar(value=100)

# ---------- Widgets ----------
tb.Label(root, text="Signature Verification System", font=("Helvetica", 18, "bold")).pack(pady=10)

frame = tb.Frame(root)
frame.pack(pady=10)

tb.Label(frame, text="Reference Signature:").grid(row=0, column=0, sticky="e", padx=10)
tb.Entry(frame, textvariable=reference_path, width=40).grid(row=0, column=1, padx=5)
tb.Button(frame, text="Browse", command=browse_reference, bootstyle="primary").grid(row=0, column=2, padx=5)

tb.Label(frame, text="Test Signature:").grid(row=1, column=0, sticky="e", padx=10, pady=10)
tb.Entry(frame, textvariable=test_path, width=40).grid(row=1, column=1, padx=5)
tb.Button(frame, text="Browse", command=browse_test, bootstyle="primary").grid(row=1, column=2, padx=5)

preview_frame = tb.Frame(root)
preview_frame.pack(pady=5)

reference_canvas = tk.Canvas(preview_frame, width=160, height=100, bg="#dee2e6", bd=2, relief="ridge")
reference_canvas.grid(row=0, column=0, padx=10)
tb.Label(preview_frame, text="Reference Preview").grid(row=1, column=0)

test_canvas = tk.Canvas(preview_frame, width=160, height=100, bg="#dee2e6", bd=2, relief="ridge")
test_canvas.grid(row=0, column=1, padx=10)
tb.Label(preview_frame, text="Test Preview").grid(row=1, column=1)

tb.Button(root, text="🔍 Verify Signature", command=verify_and_display, bootstyle="success",
          width=25).pack(pady=15)

progress_bar = tb.Progressbar(root, length=300, mode='determinate')
progress_bar.pack(pady=5)

result_label = tb.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=5)

thresh_frame = tb.Frame(root)
thresh_frame.pack(pady=10)
tb.Label(thresh_frame, text="Match Threshold (adjust if needed):").pack()
thresh_slider = tb.Scale(thresh_frame, from_=10, to=300, variable=match_threshold,
                         orient="horizontal", length=300)
thresh_slider.pack()

log_box = tk.Text(root, height=10, width=100)
log_box.pack(pady=10)

root.mainloop()