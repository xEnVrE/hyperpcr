import argparse
import os
import numpy
import time
import torch
import yarp
from pcr.model import PCRNetwork as Model
from pcr.utils import Normalize, Denormalize
from pcr.default_config import Config
from pcr.misc import download_checkpoint, download_asset


class InferenceModule (yarp.RFModule):

    def __init__(self, options):

        self.options = options
        self.options.width = 1280
        self.options.height = 720

        # Set requested GPU
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu_id)

        # Initialize YARP network
        yarp.Network.init()

        # Initialize RF module
        yarp.RFModule.__init__(self)

        # Initialize inference
        ckpt_path = download_checkpoint(f'grasping.ckpt')
        self.model = Model(config=Config.Model)
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        self.model.cuda()
        self.model.eval()

        # Initialize YARP ports
        self.depth_in = yarp.BufferedPortImageFloat()
        self.depth_in.open("/hyperpcr/depth:i")

        self.mask_in = yarp.BufferedPortImageMono()
        self.mask_in.open("/hyperpcr/mask:i")

        # Input buffers initialization
        self.depth_buffer = bytearray(numpy.zeros((self.options.height, self.options.width, 1), dtype = numpy.float32))
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(self.options.width, self.options.height)
        self.depth_image.setExternal(self.depth_buffer, self.options.width, self.options.height)

        self.mask_buffer = bytearray(numpy.zeros((self.options.height, self.options.width, 1), dtype = numpy.uint8))
        self.mask_image = yarp.ImageMono()
        self.mask_image.resize(self.options.width, self.options.height)
        self.mask_image.setExternal(self.mask_buffer, self.options.width, self.options.height)

        self.store_image_selector()


    def store_image_selector(self):

        fx = float(1229.4285612615463)
        fy = float(1229.4285612615463)
        cx = float(640)
        cy = float(360)
        self.selector_u = numpy.zeros((720, 1280), dtype = numpy.float32)
        self.selector_v = numpy.zeros((720, 1280), dtype = numpy.float32)
        for v in range(720):
            for u in range(1280):
                self.selector_u[v, u] = (u - cx) / fx
                self.selector_v[v, u] = (v - cy) / fy


    def close(self):

        self.depth_in.close()
        self.mask_in.close()

        return True


    def getPeriod(self):

        return self.options.period


    def updateModule(self):

        depth = self.depth_in.read(False)
        mask = self.mask_in.read(False)

        if (depth is not None) and (mask is not None):

            starter = torch.cuda.Event(enable_timing = True)
            ender = torch.cuda.Event(enable_timing = True)
            starter.record()

            self.depth_image.copy(depth)
            depth_frame = numpy.frombuffer(self.depth_buffer, dtype=numpy.float32).reshape(self.options.height, self.options.width)

            self.mask_image.copy(mask)
            mask_frame = numpy.frombuffer(self.mask_buffer, dtype=numpy.uint8).reshape(self.options.height, self.options.width)

            mask_selector = mask_frame != 0
            depth_valid_selector = depth_frame < 0.55
            valid_selector = mask_selector & depth_valid_selector
            z = depth_frame[valid_selector]
            x_z = self.selector_u[valid_selector]
            y_z = self.selector_v[valid_selector]

            cloud = numpy.zeros((z.shape[0], 3), dtype = numpy.float32)
            cloud[:, 0] = x_z * z
            cloud[:, 1] = y_z * z
            cloud[:, 2] = z

            partial, ctx = Normalize(Config.Processing)(cloud)
            partial = torch.tensor(partial, dtype=torch.float32).cuda().unsqueeze(0)
            complete, probabilities = self.model(partial)
            complete = complete.squeeze(0).cpu().numpy()
            complete = Denormalize(Config.Processing)(complete, ctx)

            ender.record()
            torch.cuda.synchronize()
            elapsed = starter.elapsed_time(ender) / 1000.0
            print(1.0 / elapsed)

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type = int, default = '0')
    parser.add_argument('--period', type = float, default = 0.016)

    options = parser.parse_args()

    module = InferenceModule(options)
    module.runModule()
