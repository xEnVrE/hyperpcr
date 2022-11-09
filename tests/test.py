import numpy as np
import torch

from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.default_config import Config
from pcr.misc_download import download_checkpoint, download_asset


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
    partial_in, ctx = Normalize(Config.Processing)(partial)
    partial_in = torch.tensor(partial_in, dtype=torch.float32).cuda().unsqueeze(0)

    complete, probabilities = model(partial_in)
    complete = complete.squeeze(0).cpu().numpy()
    complete = Denormalize(Config.Processing)(complete, ctx)

    np.savetxt('./reconstruction.txt', complete)

    try:
        import open3d as o3d
        o3d.visualization.draw([
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(partial)).paint_uniform_color([0, 0, 1]),
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(complete)).paint_uniform_color([0, 1, 1]),
        ])
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
