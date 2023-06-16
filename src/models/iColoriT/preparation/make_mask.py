import argparse
import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm


def make_fixed_hint():
    parser = argparse.ArgumentParser(description="Making fixed hint set for interactive colorization")
    parser.add_argument('--img_dir', type=str, default='../data/imgs')
    parser.add_argument('--hint_dir', type=str, default='../data/hints')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--hint_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()

    assert args.img_size % args.hint_size == 0
    filenames = sorted(os.listdir(args.img_dir))
    random.seed(args.seed)

    for hint_ratio in [0, 1, 2, 5, 50]:
        total_pixels = (args.img_size // args.hint_size - 1) ** 2
        #total_pixels = (args.img_size - 1) ** 2
        num_hint = int(total_pixels * hint_ratio * 0.01)
        print(f'ratio {hint_ratio * 0.01}, no. hints: {num_hint}')
        for file in tqdm(filenames):
            lines = [f'{random.randint(0, args.img_size//args.hint_size - 1) * args.hint_size} '
                     f'{random.randint(0, args.img_size//args.hint_size - 1) * args.hint_size}\n'
                     for _ in range(num_hint)]
            txt_file = osp.join(args.hint_dir, str(args.seed), f'h{args.hint_size}-n{hint_ratio}',
                                osp.splitext(file)[0] + '.txt')
            os.makedirs(osp.dirname(txt_file), exist_ok=True)
            with open(txt_file, 'w') as f:
                f.writelines(lines)


if __name__ == '__main__':
    make_fixed_hint()
    #make_fixed_hint_percentage()