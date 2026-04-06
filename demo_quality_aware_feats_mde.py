import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.transforms as T

from networks.multi_distortion_encoder import MultiDistortionEncoder


PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path: str, device: torch.device) -> MultiDistortionEncoder:
    model = MultiDistortionEncoder()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
        state = {
            k.replace('module.encoder_q.', '').replace('encoder_q.', ''): v
            for k, v in state.items()
            if 'encoder_q' in k
        }
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    print(f'Loaded checkpoint from: {checkpoint_path}')
    return model


@torch.no_grad()
def extract_features(
    model: MultiDistortionEncoder,
    image_paths: list,
    device: torch.device,
    batch_size: int = 32,
    show_diagnosis: bool = False,
) -> np.ndarray:
    all_features = []
    distortion_types = model.DISTORTION_TYPES

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        batch_tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_tensors.append(PREPROCESS(img))
            except Exception as e:
                print(f'Warning: could not load {path}: {e}')
                batch_tensors.append(torch.zeros(3, 224, 224))

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        embeddings, weights = model.forward_with_weights(batch)

        all_features.append(embeddings.cpu().numpy())

        if show_diagnosis:
            w = weights.cpu().numpy()
            for j, path in enumerate(batch_paths):
                diag = {
                    dtype: f'{w[j, k]:.3f}'
                    for k, dtype in enumerate(distortion_types)
                }
                print(f'{Path(path).name}: {diag}')

        if (i // batch_size) % 10 == 0:
            print(f'Processed {min(i + batch_size, len(image_paths))}'
                  f'/{len(image_paths)} images')

    return np.vstack(all_features)


def main():
    parser = argparse.ArgumentParser('Extract MDE quality features')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained MDE checkpoint (.pth)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='CSV with image paths (same format as Re-IQA)')
    parser.add_argument('--output_path', type=str,
                        default='features/quality_feats_mde.npy',
                        help='Where to save the extracted features')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--show_diagnosis', action='store_true',
                        help='Print distortion type weights per image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = load_model(args.checkpoint, device)

    import pandas as pd
    df = pd.read_csv(args.csv_path)
    image_paths = df.iloc[:, 0].tolist()
    print(f'Extracting features for {len(image_paths)} images...')

    features = extract_features(
        model, image_paths, device,
        batch_size=args.batch_size,
        show_diagnosis=args.show_diagnosis,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, features)
    print(f'Saved features: {features.shape} -> {args.output_path}')
    print()
    print('Next step: concatenate with content features and train regressor.')
    print('  quality = np.load("features/quality_feats_mde.npy")')
    print('  content = np.load("features/content_feats.npy")')
    print('  combined = np.concatenate([quality, content], axis=1)')


if __name__ == '__main__':
    main()
