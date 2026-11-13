import argparse
import csv
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

METHODS = ('baseline', 'improved')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
METRIC_NAMES = ('accuracy', 'precision', 'recall', 'iou', 'f1', 'specificity', 'sensitivity')


def load_photo(path):
    with Image.open(path) as source:
        # Open the image and correct its orientation using EXIF information
        rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    # Convert RGB to BGR because OpenCV uses BGR format
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Encode the image in PNG format
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f'Could not save {path}')
    # Save the encoded image to the file
    path.write_bytes(encoded.tobytes())


def red_mask_hsv(photo):
    # Convert the BGR image to HSV color space
    hsv = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    # Create masks for the lower and upper red color ranges
    lower_red = cv2.inRange(hsv, (0, 85, 65), (12, 255, 255))
    upper_red = cv2.inRange(hsv, (170, 85, 65), (179, 255, 255))
    # Combine the two masks to get the final red mask
    return cv2.bitwise_or(lower_red, upper_red)


def segment(photo, method):
    mask = red_mask_hsv(photo)
    if method == 'baseline':
        return mask
    if method == 'improved':
        # Remove isolated noise using a 5x5 median filter
        mask = cv2.medianBlur(mask, 5)
        # Create a 5x5 elliptical kernel for morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # Apply morphological opening and closing to refine the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    raise ValueError(f'Unknown method: {method}')


def evaluate(predicted, reference):
    # Convert predicted and reference masks to boolean arrays where True indicates a positive (red) pixel
    p, r = predicted > 0, reference > 0
    # TP: red pixels correctly detected as red
    tp = np.count_nonzero(p & r)
    # TN: background pixels correctly detected as background
    tn = np.count_nonzero(~p & ~r)
    # FP: background pixels incorrectly detected as red
    fp = np.count_nonzero(p & ~r)
    # FN: red pixels incorrectly detected as background
    fn = np.count_nonzero(~p & r)
    def ratio(a, b):
        return float(a / b) if b else 0.0
    return {
        # Accuracy: percentage of all correctly classified pixels
        'accuracy': ratio(tp + tn, tp + tn + fp + fn),
        # Precision: how many detected red pixels are actually red
        'precision': ratio(tp, tp + fp),
        # Recall: how many actual red pixels were detected
        'recall': ratio(tp, tp + fn),
        # IoU: overlap between predicted and ground truth masks
        'iou': ratio(tp, tp + fp + fn),
        # F1-score: balance between precision and recall
        'f1': ratio(2 * tp, 2 * tp + fp + fn),  
        'specificity': ratio(tn, tn+fp),
        'sensitivity': ratio(tp, tp+fn),
    }


def draw_boxes(photo, mask):
    marked = photo.copy()
    # Find external contours of detected white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Ignore regions smaller than a minimum area
    # The threshold depends partly on image size
    minimum_area = max(100, photo.shape[0] * photo.shape[1] * 0.00015)
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) < minimum_area:
            continue
        # Find a rectangular bounding box around the region
        x, y, w, h = cv2.boundingRect(contour)
        # Draw a green rectangle on the photo
        cv2.rectangle(
            marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
    return marked, count


def comparison(photo, reference, baseline, improved):
    h, w = photo.shape[:2]
    # Resize images to fit within 420 pixels while keeping proportions
    scale = min(1.0, 420 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    panels = [('Original', photo), ('Ground truth', reference),
              ('Baseline', baseline), ('Improved', improved)]
    result = []
    for label, panel in panels:
        # Convert grayscale masks to BGR for displaying together
        if panel.ndim == 2:
            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        # Nearest-neighbor keeps binary mask values unchanged
        thumb = cv2.resize(panel, size, interpolation=cv2.INTER_NEAREST if label != 'Original' else cv2.INTER_AREA)
        frame = np.full((size[1] + 35, size[0], 3), 255, dtype=np.uint8)
        frame[:size[1], :] = thumb
        cv2.putText(frame, label, (8, size[1] + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (20, 20, 20), 1, cv2.LINE_AA)
        result.append(frame)
    return cv2.hconcat(result)


def write_csv(path, records, columns):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def run(images_dir, masks_dir, output_dir, repetitions, resize_masks=False):
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Missing folder: {images_dir}')
    if not masks_dir.is_dir():
        raise FileNotFoundError(f'Missing folder: {masks_dir}')
    photos = sorted((p for p in images_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                    key=lambda p: p.name.lower())
    if not photos:
        raise ValueError('No photographs found in real_images/')
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in photos:
        photo = load_photo(path)
        mask_path = masks_dir / (path.stem + '.png')
        if not mask_path.is_file():
            raise FileNotFoundError(f'Ground truth mask missing: {mask_path}')
        try:
            with Image.open(mask_path) as source:
                reference = np.asarray(source.convert('L'))
        except Exception as exc:
            raise ValueError(f'Cannot open mask: {mask_path}') from exc
        if reference.shape != photo.shape[:2]:
            image_size = (photo.shape[1], photo.shape[0])
            mask_size = (reference.shape[1], reference.shape[0])
            if not resize_masks:
                raise ValueError(
                    f'Mask size mismatch for {path.name}: image={image_size}, '
                    f'mask={mask_size}. Use --resize-masks only if both '
                    f'files show the exact same uncropped scene.'
                )
            print(f'WARNING: resizing {mask_path.name} from {mask_size} to {image_size}. '
                  'Check mask alignment before using metrics in the report.')
            reference = cv2.resize(reference, image_size, interpolation=cv2.INTER_NEAREST)
        if not np.all((reference == 0) | (reference == 255)):
            raise ValueError(f'Mask must contain only black (0) and white (255): {mask_path}')

        predicted = {}
        for method in METHODS:
            segment(photo, method)  # warm-up
            times = []
            for _ in range(repetitions):
                start = time.perf_counter()
                mask = segment(photo, method)
                times.append((time.perf_counter() - start) * 1000)
            predicted[method] = mask
            boxes, detections = draw_boxes(photo, mask)
            save_png(output_dir / f'{path.stem}_{method}_mask.png', mask)
            save_png(output_dir / f'{path.stem}_{method}_boxes.png', boxes)
            records.append({
                'image': path.name, 'method': method,
                **evaluate(mask, reference),
                'time_ms': float(np.median(times)),
                'detections': detections,
            })
        save_png(output_dir / f'{path.stem}_comparison.png',
                 comparison(photo, reference, predicted['baseline'], predicted['improved']))
        print(f'Processed {path.name}')

    columns = ['image', 'method', *METRIC_NAMES, 'time_ms', 'detections']
    write_csv(output_dir / 'results.csv', records, columns)
    summary = []
    for method in METHODS:
        group = [record for record in records if record['method'] == method]
        row = {'method': method, 'images': len(group)}
        for metric in (*METRIC_NAMES, 'time_ms', 'detections'):
            row[metric] = float(np.mean([record[metric] for record in group]))
        summary.append(row)
        print(f"{method:9} IoU={row['iou']:.4f}  F1={row['f1']:.4f} sensivity={row['sensitivity']:.4f} specificity={row['specificity']:.4f}  "
              f"Time={row['time_ms']:.2f} ms")
    write_csv(output_dir / 'summary.csv', summary,
              ['method', 'images', *METRIC_NAMES, 'time_ms', 'detections'])
    print(f'Results saved in: {output_dir.resolve()}')


def main():
    parser = argparse.ArgumentParser(description='Compare two red-region segmentation methods')
    parser.add_argument('--images', type=Path, default=Path('real_images'))
    parser.add_argument('--masks', type=Path, default=Path('real_masks'))
    parser.add_argument('--output', type=Path, default=Path('results_real'))
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--resize-masks', action='store_true',
                        help='Resize mismatched masks using nearest neighbor (verify alignment)')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('--repetitions must be at least 1')
    run(args.images, args.masks, args.output, args.repetitions, args.resize_masks)


if __name__ == '__main__':
    main()
