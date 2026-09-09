from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class SquarePad:
    """Pad an image to a square."""
    def __call__(self, image):
        w, h = image.size
        m = max(w, h)
        left = (m - w) // 2
        top = (m - h) // 2
        right = m - w - left
        bottom = m - h - top
        return ImageOps.expand(image, border=(left, top, right, bottom), fill=0)

def build_transform(image_size=224, preserve_full_frame=False, train=False):
    """Build standardized torchvision transforms for CLIP models."""
    if preserve_full_frame:
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            SquarePad(),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    
    if train:
        return transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(image_size + 32, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
        
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])