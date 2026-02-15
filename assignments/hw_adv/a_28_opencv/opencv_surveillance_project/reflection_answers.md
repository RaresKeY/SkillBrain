# Reflection Answers

## Exercise 1

### 1. What happens if you set `area > 100` instead of `area > 500`?
It becomes more sensitive and detects much smaller blue regions, but false positives increase because noise and tiny artifacts are no longer filtered out.

### 2. Challenge: Modify the code to detect RED objects. What changes?
Use HSV ranges for red, which wraps around hue boundaries. Typically combine two masks: low red (`H` near 0) and high red (`H` near 180), then merge them.

## Exercise 2

### 1. What else can we clean beyond old writings?
We can improve receipts, whiteboard photos, scanned contracts, license plates, and forms before OCR or archival.

### 2. Real-world application: How could this help with OCR?
Adaptive thresholding improves foreground/background separation under uneven lighting, which reduces OCR misreads and improves character segmentation.

## Exercise 3

### 1. Would your algorithm hold for 1000 images per second?
Not without optimization. It would need batching, GPU acceleration, lower resolution, parallel processing, and stricter ROI filtering to sustain that throughput.

### 2. How would you identify circles vs rectangles using contours?
Approximate polygons with `cv2.approxPolyDP`: ~4 vertices suggests rectangles. For circles, compare contour area to enclosing-circle area and circularity (`4πA/P²`) near 1.

## Final Project

### 1. What makes smoke detection harder than motion detection?
Motion is a direct frame-difference signal. Smoke is diffuse, semi-transparent, lighting-sensitive, and visually similar to fog/steam/bright backgrounds, so it needs temporal pattern analysis.

### 2. How would you reduce false alarms (e.g., shadows, lighting changes)?
Use background subtraction with adaptation, edge/texture features, region-of-interest masking, temporal smoothing, and multi-condition alerts (color + growth + upward drift).

### 3. Your system catches an intruder at 3 AM. What additional features would make it more useful?
Auto-notifications (mobile/email), clip recording before/after event, person detection, siren integration, and remote live feed access.

### 4. Ethical considerations of surveillance systems?
Privacy, consent, retention limits, secure storage, access control, bias in detection, and transparent use policies are critical.
