"""
FLASH: Efficient Impact Fall Detection
IEEE ICIP 2026 - Official Implementation
Usage:
  python run.py --mode train
  python run.py --mode test
  python run.py --mode ablation
  python run.py --mode occlusion
  python run.py --mode all
"""

import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(
        description='FLASH - Impact Fall Detection (IEEE ICIP 2026)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'test', 'ablation', 'occlusion', 'all'],
        help='Mode to run: train | test | ablation | occlusion | all'
    )
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    args = parser.parse_args()

    print("=" * 60)
    print("FLASH: Efficient Impact Fall Detection")
    print("IEEE ICIP 2026 - Official Implementation")
    print("=" * 60)

    if args.mode in ['train', 'all']:
        print("\n[1/4] Running main training...")
        subprocess.run([sys.executable, 'train.py',
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay)
        ])

    if args.mode in ['ablation', 'all']:
        print("\n[2/4] Running hyperedge ablation study...")
        subprocess.run([sys.executable, 'train_ablation.py'])

    if args.mode in ['occlusion', 'all']:
        print("\n[3/4] Running occlusion robustness test...")
        subprocess.run([sys.executable, 'test_occlusion.py'])

    if args.mode in ['all']:
        print("\n[4/4] All experiments completed!")
        print("Results saved in training_results/")

    print("\nDone!")

if __name__ == '__main__':
    main()
