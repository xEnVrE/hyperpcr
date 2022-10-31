import numpy as np
import torch

from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.default_config import Config
from pcr.misc import download_checkpoint, download_asset


def main():

    ckpt_path = download_checkpoint(f'grasping.ckpt')
    asset_path = download_asset(f'partial_bleach_317.npy')

    print('Checkpoint downloaded in ' + ckpt_path)
    print('Asset downloaded in ' + asset_path)

    model = Model(config=Config.Model)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.cuda()
    model.eval()

    partial = np.load(asset_path)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    while True:
        starter.record()

        partial_in, ctx = Normalize(Config.Processing)(partial)

        partial_in = torch.tensor(partial_in, dtype=torch.float32).cuda().unsqueeze(0)
        complete, probabilities = model(partial_in)

        complete = complete.squeeze(0).cpu().numpy()
        complete = Denormalize(Config.Processing)(complete, ctx)

        ender.record()
        torch.cuda.synchronize()

        elapsed = starter.elapsed_time(ender) / 1000.0
        print('{} Hz'.format(1.0 / elapsed))


if __name__ == '__main__':
    main()
