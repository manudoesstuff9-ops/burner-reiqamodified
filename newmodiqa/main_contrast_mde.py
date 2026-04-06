import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from moco.builder_mde import MoCo_MDE
from moco.losses import MultiDistortionLoss
from moco.distortion_augmentations import (
    DistortionAugmentPair,
    ManifoldTripletTransform,
    DISTORTION_FN,
)

import moco.optimizer
from datasets.dataset import ImageFolderInstance


def get_args():
    parser = argparse.ArgumentParser('Re-IQA + ARNIQA Multi-Distortion Training')

    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='./expt_mde')
    parser.add_argument('--tb_path', type=str, default='./expt_mde')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=6.0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--optimizer', type=str, default='LARS')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist-url', type=str, default='tcp://localhost:10001')

    parser.add_argument('--moco_k', type=int, default=65536,
                        help='Queue size for MoCo negatives')
    parser.add_argument('--moco_m', type=float, default=0.999,
                        help='Momentum for key encoder update')
    parser.add_argument('--moco_t', type=float, default=0.2,
                        help='Temperature for InfoNCE loss')

    parser.add_argument('--lambda_triplet', type=float, default=0.5,
                        help='Weight for manifold triplet loss. '
                             'Higher = stronger severity ordering signal. '
                             'Range: [0.1, 1.0], start with 0.5.')
    parser.add_argument('--lambda_gate', type=float, default=0.1,
                        help='Weight for gating entropy regularisation. '
                             'Prevents head collapse. Keep small: [0.05, 0.2].')

    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training from.')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs.')

    return parser.parse_args()


class MultiDistortionDataset(torch.utils.data.Dataset):

    def __init__(self, base_dataset, patch_size: int = 224):
        self.base = base_dataset
        self.patch_size = patch_size

        from moco.distortion_augmentations import get_base_transform
        self.pair_transform = get_base_transform(patch_size)

        self.triplet_transforms = {
            name: ManifoldTripletTransform(name, patch_size=patch_size)
            for name in DISTORTION_FN.keys()
        }

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = item[0] if isinstance(item, (list, tuple)) else item

        view1 = self.pair_transform(img)
        view2 = self.pair_transform(img)

        triplets = {}
        for name, transform in self.triplet_transforms.items():
            anchor, positive, negative = transform(img)
            triplets[name] = (anchor, positive, negative)

        return view1, view2, triplets


def train_step(model, criterion, optimizer, batch, device, distortion_idx=0):
    view1, view2, triplets = batch

    im_q = view1.to(device)
    im_k = view2.to(device)

    moco = model.module if hasattr(model, 'module') else model
    q, k, queue, gate_weights_q = moco(im_q, im_k)

    distortion_types = list(triplets.keys())
    dist_type = distortion_types[distortion_idx % len(distortion_types)]

    anc_imgs, pos_imgs, neg_imgs = triplets[dist_type]
    anc_imgs = anc_imgs.to(device)
    pos_imgs = pos_imgs.to(device)
    neg_imgs = neg_imgs.to(device)

    feat_a = moco.encoder_q._get_backbone_features(anc_imgs)
    feat_p = moco.encoder_q._get_backbone_features(pos_imgs)
    feat_n = moco.encoder_q._get_backbone_features(neg_imgs)

    head = moco.encoder_q.heads[dist_type]
    anchor_embed = head(feat_a)
    positive_embed = head(feat_p)
    negative_embed = head(feat_n)

    loss, breakdown = criterion(
        q=q,
        k=k,
        queue=queue,
        anchor=anchor_embed,
        positive=positive_embed,
        negative=negative_embed,
        gate_weights=gate_weights_q,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return breakdown


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MoCo_MDE(
        embed_dim=128,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
    ).to(device)

    if args.multiprocessing_distributed:
        model = nn.parallel.DistributedDataParallel(model)

    criterion = MultiDistortionLoss(
        temperature=args.moco_t,
        lambda_triplet=args.lambda_triplet,
        lambda_gate=args.lambda_gate,
    ).to(device)

    if args.optimizer == 'LARS':
        optimizer = moco.optimizer.LARS(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4,
            momentum=0.9,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed at epoch {start_epoch}')

    from datasets.dataset import IQAImageClass
    
    print(f'Loading dataset from: {args.csv_path}')
    base_dataset = IQAImageClass(
        csv_path=args.csv_path,
        n_aug=2,
        n_scale=1,
        n_distortions=1,
        patch_size=args.patch_size,
    )
    
    dataset = MultiDistortionDataset(base_dataset, args.patch_size)
    
    sampler = DistributedSampler(dataset) if args.multiprocessing_distributed else None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f'Starting training for {args.epochs} epochs')
    print(f'  batch_size     = {args.batch_size}')
    print(f'  learning_rate  = {args.learning_rate}')
    print(f'  lambda_triplet = {args.lambda_triplet}')
    print(f'  lambda_gate    = {args.lambda_gate}')
    print(f'  Total batches per epoch = {len(loader)}')

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        epoch_losses = {}

        for batch_idx, batch in enumerate(loader):
            breakdown = train_step(model, criterion, optimizer, batch, device, batch_idx)
            
            for k, v in breakdown.items():
                if k not in epoch_losses:
                    epoch_losses[k] = 0
                epoch_losses[k] += v

            if batch_idx % 10 == 0:
                avg_loss = {k: v / (batch_idx + 1) for k, v in epoch_losses.items()}
                print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {avg_loss}')

        print(f'Epoch {epoch} finished. Avg loss: {epoch_losses}')

        if (epoch + 1) % args.save_freq == 0:
            os.makedirs(args.model_path, exist_ok=True)
            save_path = os.path.join(args.model_path, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, save_path)
            print(f'Saved checkpoint: {save_path}')

    print('Training completed!')


if __name__ == '__main__':
    main()
