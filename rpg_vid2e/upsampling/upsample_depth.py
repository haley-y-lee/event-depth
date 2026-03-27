import argparse
import os
# Must be set before importing torch.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import upsampler_depth


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--timestamps_file", required=True, help="Path to reference timestamps.txt")
    return parser.parse_args()



def main():
    flags = get_flags()
    upsampler = Upsampler(flags.input_dir, flags.output_dir, flags.timestamps_file)
    upsampler.upsample()



if __name__ == '__main__':
    main()