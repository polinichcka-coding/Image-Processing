#python create_masks.py --images new_images --masks real_masks
#py create_masks.py
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.avif'}


def load_bgr(path):
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_png(path, mask):
    ok, encoded = cv2.imencode('.png', mask)
    if not ok:
        raise OSError(f'Could not encode mask: {path}')
    path.write_bytes(encoded.tobytes())


def annotate(path, masks_dir, max_side=1100):
    original = load_bgr(path)
    h, w = original.shape[:2]
    target = masks_dir / f'{path.stem}.png'
    if target.exists():
        with Image.open(target) as im:
            mask = np.asarray(im.convert('L')).copy()
        if mask.shape != (h, w):
            raise ValueError(f'Existing mask has wrong size: {target}')
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    else:
        mask = np.zeros((h, w), np.uint8)

    scale = min(1.0, max_side / max(h, w))
    display_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    brush = 12
    drawing = False
    paint_value = 255
    last = None
    title = 'Ground truth editor'

    def mouse(event, x, y, flags, userdata):
        nonlocal drawing, paint_value, last
        px = min(w - 1, max(0, round(x / scale)))
        py = min(h - 1, max(0, round(y / scale)))
        point = (px, py)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            drawing = True
            paint_value = 255 if event == cv2.EVENT_LBUTTONDOWN else 0
            cv2.circle(mask, point, brush, paint_value, -1)
            last = point
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(mask, last, point, paint_value, brush * 2, cv2.LINE_8)
            cv2.circle(mask, point, brush, paint_value, -1)
            last = point
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
            last = None

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, mouse)
    while True:
        preview = original.copy()
        red_overlay = np.zeros_like(preview)
        red_overlay[:, :] = (30, 30, 245)
        region = mask > 0
        preview[region] = cv2.addWeighted(preview, 0.45, red_overlay, 0.55, 0)[region]
        preview = cv2.resize(preview, display_size, interpolation=cv2.INTER_AREA)
        caption = f'{path.name} | brush {brush}px | L paint R erase | +/- size | S save N skip R reset Q quit'
        cv2.putText(preview, caption[:max(20, display_size[0] // 8)], (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(preview, caption[:max(20, display_size[0] // 8)], (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow(title, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('s'):
            save_png(target, mask)
            print(f'Saved: {target}')
            cv2.destroyWindow(title)
            return True
        if key == ord('n'):
            print(f'Skipped: {path.name}')
            cv2.destroyWindow(title)
            return True
        if key == ord('r'):
            mask[:] = 0
        if key in (ord('+'), ord('=')):
            brush = min(200, brush + 3)
        if key in (ord('-'), ord('_')):
            brush = max(1, brush - 3)
        if key in (ord('q'), 27):
            cv2.destroyWindow(title)
            return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('real_images'))
    parser.add_argument('--masks', type=Path, default=Path('real_masks'))
    parser.add_argument('--max-side', type=int, default=1100)
    args = parser.parse_args()
    if not args.images.is_dir():
        parser.error(f'Image directory not found: {args.images}')
    if args.max_side < 100:
        parser.error('--max-side must be at least 100')
    paths = sorted((p for p in args.images.iterdir()
                    if p.is_file() and p.suffix.lower() in EXTENSIONS),
                   key=lambda p: p.name.lower())
    if not paths:
        parser.error('No supported images found')
    args.masks.mkdir(parents=True, exist_ok=True)
    print('Paint ONLY the actual red regions of the sign, not the entire sign.')
    print('Left mouse: paint; right mouse: erase; +/-: brush; S: save; N: skip; Q: quit.')
    for path in paths:
        if not annotate(path, args.masks, args.max_side):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
