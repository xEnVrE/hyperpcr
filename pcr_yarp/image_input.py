import numpy
import robotsio
import yarp


class ImageInput():

    def __init__(self, prefix, config):

        self.config = config

        # Initialize YARP ports
        if config.Input.joint_input_mode:
            self.images_in = robotsio.BufferedPortYarpImageOfMonoFloat()
            self.images_in.open('/' + prefix + '/images:i')
        else:
            self.depth_in = yarp.BufferedPortImageFloat()
            self.depth_in.open('/' + prefix + '/depth:i')

            self.mask_in = yarp.BufferedPortImageMono()
            self.mask_in.open('/' + prefix + '/mask:i')

        # Input buffers initialization
        self.depth_buffer = bytearray(numpy.zeros((self.config.Camera.height, self.config.Camera.width, 1), dtype = numpy.float32))
        self.depth_image = yarp.ImageFloat()
        self.depth_image.resize(self.config.Camera.width, self.config.Camera.height)
        self.depth_image.setExternal(self.depth_buffer, self.config.Camera.width, self.config.Camera.height)

        self.mask_buffer = bytearray(numpy.zeros((self.config.Camera.height, self.config.Camera.width, 1), dtype = numpy.uint8))
        self.mask_image = yarp.ImageMono()
        self.mask_image.resize(self.config.Camera.width, self.config.Camera.height)
        self.mask_image.setExternal(self.mask_buffer, self.config.Camera.width, self.config.Camera.height)


    def get_images(self):

        depth = None
        mask = None

        if self.config.Input.joint_input_mode:
            images_in = self.images_in.read(False)
            if images_in is not None:
                depth = images_in.image_float
                mask = images_in.image_mono
        else:
            depth = self.depth_in.read(False)
            mask = self.mask_in.read(False)

        if (depth is not None) and (mask is not None):

            self.depth_image.copy(depth)
            depth_frame = numpy.frombuffer(self.depth_buffer, dtype=numpy.float32).reshape(self.config.Camera.height, self.config.Camera.width)

            self.mask_image.copy(mask)
            mask_frame = numpy.frombuffer(self.mask_buffer, dtype=numpy.uint8).reshape(self.config.Camera.height, self.config.Camera.width)

            return True, depth_frame, mask_frame
        else:
            return False, None, None


    def close(self):

        if self.config.Input.joint_input_mode:
            self.images_in.close()
        else:
            self.depth_in.close()
            self.mask_in.close()
