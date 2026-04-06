import io
import random
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T


def get_base_transform(patch_size: int = 224) -> T.Compose:
    
    return T.Compose([
        T.RandomResizedCrop(patch_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def apply_gaussian_noise(img: Image.Image, severity: float) -> Image.Image:
    
    sigma = 2.0 + severity * 58.0
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def apply_blur(img: Image.Image, severity: float) -> Image.Image:
    
    radius = 0.3 + severity * 7.7
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(img: Image.Image, severity: float) -> Image.Image:
    
    quality = max(5, int(95 - severity * 90))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def apply_weather_haze(img: Image.Image, severity: float) -> Image.Image:
    
    alpha = 0.05 + severity * 0.65
    arr = np.array(img, dtype=np.float32)
    hazy = arr * (1.0 - alpha) + 255.0 * alpha
    return Image.fromarray(np.clip(hazy, 0, 255).astype(np.uint8))


DISTORTION_FN = {
    'gaussian': apply_gaussian_noise,
    'blur':     apply_blur,
    'jpeg':     apply_jpeg_compression,
    'weather':  apply_weather_haze,
}
#  SECTION 3: DistortionAugmentPair
#  Creates the two views needed for standard MoCo contrastive
#  learning, but using distortion-specific augmentations.
#
#  Used during Phase 1 training of each specialist head.
# ─────────────────────────────────────────────────────────────

class DistortionAugmentPair:
    """
    Creates two distorted views of an image for contrastive learning.

    Both views have the same distortion TYPE but different severities,
    randomly sampled from a specified range. This teaches the head
    that "similar severity = similar embedding."

    Args:
        distortion_type : one of 'gaussian', 'blur', 'jpeg', 'weather'
        severity_range  : (min, max) severity for BOTH views.
                          Default (0.1, 0.9) covers the full range
                          while avoiding the trivial extremes.
        patch_size      : output image size for base transform

    Usage:
        transform = DistortionAugmentPair('blur')
        view1, view2 = transform(pil_image)
        # Both views are blurred, at randomly chosen similar severities
    """

    def __init__(
        self,
        distortion_type: str,
        severity_range: tuple = (0.1, 0.9),
        patch_size: int = 224,
    ):
        assert distortion_type in DISTORTION_FN, \
            f"Unknown distortion type '{distortion_type}'. " \
            f"Choose from: {list(DISTORTION_FN.keys())}"

        self.distort_fn = DISTORTION_FN[distortion_type]
        self.severity_range = severity_range
        self.base_transform = get_base_transform(patch_size)

    def __call__(self, img: Image.Image):
        """
        Args:
            img: PIL Image (clean or already slightly degraded)
        Returns:
            (view1, view2): two augmented tensor views
        """
        s1 = random.uniform(*self.severity_range)
        s2 = random.uniform(*self.severity_range)

        view1 = self.base_transform(self.distort_fn(img, s1))
        view2 = self.base_transform(self.distort_fn(img, s2))

        return view1, view2


# ─────────────────────────────────────────────────────────────
#  SECTION 4: ManifoldTripletTransform
#  Creates anchor / positive / negative triplets for
#  ARNIQA-style manifold training.
#
#  anchor   = clean image (severity 0)
#  positive = mildly distorted  (severity in LOW range)
#  negative = heavily distorted (severity in HIGH range)
#
#  The triplet loss pushes positive close to anchor and
#  negative far away — creating the smooth severity manifold.
# ─────────────────────────────────────────────────────────────

class ManifoldTripletTransform:
    """
    Creates (anchor, positive, negative) triplets for manifold learning.

    This is the ARNIQA contribution: instead of only teaching the
    network about distortion type, we also teach it about severity
    ordering. After training, "distance = severity difference" holds
    in each head's embedding space.

    Args:
        distortion_type : distortion type for this head
        low_range  : severity range for mild distortion (positive)
        high_range : severity range for heavy distortion (negative)
        patch_size : output image size

    Usage:
        triplet = ManifoldTripletTransform('gaussian')
        anchor, positive, negative = triplet(pil_image)
    """

    def __init__(
        self,
        distortion_type: str,
        low_range: tuple = (0.0, 0.3),    # mild: anchor stays clean or near-clean
        high_range: tuple = (0.6, 1.0),   # heavy: clearly degraded
        patch_size: int = 224,
    ):
        assert distortion_type in DISTORTION_FN
        self.distort_fn = DISTORTION_FN[distortion_type]
        self.low_range = low_range
        self.high_range = high_range
        self.base_transform = get_base_transform(patch_size)

    def __call__(self, img: Image.Image):
        """
        Args:
            img: clean PIL Image
        Returns:
            (anchor, positive, negative): three tensor views
            anchor   — clean or very mild distortion
            positive — mild distortion (should be close to anchor in embedding)
            negative — heavy distortion (should be far from anchor)
        """
        s_anchor   = random.uniform(0.0, 0.1)         # nearly clean
        s_positive = random.uniform(*self.low_range)
        s_negative = random.uniform(*self.high_range)

        anchor   = self.base_transform(self.distort_fn(img, s_anchor))
        positive = self.base_transform(self.distort_fn(img, s_positive))
        negative = self.base_transform(self.distort_fn(img, s_negative))

        return anchor, positive, negative
