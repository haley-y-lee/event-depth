import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname, imgs_name
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str, timestamps_file : str):
        assert os.path.isdir(input_dir), 'The input directory must exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir
#         path = os.path.join(os.path.dirname(__file__), " /home/yl3836/mono_event/rpg_vid2e/pretrained_models/film_net/Style/saved_model")
        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)

        with open(timestamps_file, 'r') as f:
            self.reference_timestamps = [float(line.strip()) for line in f.readlines()]

    def upsample(self):
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath, dirname=imgs_name)
            if sequence is None:
                continue
            sequence_counter += 1
            print('Processing sequence number {}'.format(src_absdirpath))
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_name)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)

    # def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
    #     os.makedirs(dest_imgs_dir, exist_ok=True)
    #     timestamps_list = list()

    #     idx = 0
    #     for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
    #         I0 = img_pair[0][None]
    #         I1 = img_pair[1][None]
    #         t0, t1 = time_pair

    #         total_frames, total_timestamps = self._upsample_adaptive(I0, I1, t0, t1)
    #         total_frames = [I0[0]] + total_frames
    #         timestamps = [t0] + total_timestamps

    #         sorted_indices = np.argsort(timestamps)
    #         total_frames = [total_frames[j] for j in sorted_indices]
    #         timestamps = [timestamps[i] for i in sorted_indices]

    #         timestamps_list += timestamps
    #         for frame in total_frames:
    #             # breakpoint()
    #             frame = frame[:260, :346]   # crop back to original size
    #             self._write_img(frame, idx, dest_imgs_dir)
    #             idx += 1

    #     # TODO CROP IMAGE BACK TO ORIGINAL DIMENSION
    #     # breakpoint()

    #     timestamps_list.append(t1)
    #     # self._write_img(I1[0, ...], idx, dest_imgs_dir)
    #     self._write_img(I1[0, ...][:260, :346], idx, dest_imgs_dir)       # crop back to original size
    #     self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        reference_ts = self.reference_timestamps
        timestamps_list = []
        idx = 0

        # 전체 프레임 쌍과 타임스탬프 쌍 얻기
        frames, time_pairs = sequence.get_all_pairs()

        for t in tqdm(reference_ts, desc="Upsampling Depth"):
            for (img_pair, (t0, t1)) in zip(frames, time_pairs):
                if t0 <= t <= t1:
                    I0 = img_pair[0][None]
                    I1 = img_pair[1][None]

                    # Grayscale → RGB 변환
                    if I0.shape[-1] == 1:
                        I0 = np.repeat(I0, 3, axis=-1)
                    if I1.shape[-1] == 1:
                        I1 = np.repeat(I1, 3, axis=-1)

                    # t가 정확히 경계에 있는 경우 예외 처리
                    if t == t0:
                        img = I0
                    elif t == t1:
                        img = I1
                    else:
                        dt = np.array([(t - t0) / (t1 - t0)], dtype=np.float32)
                        img, _, _ = self.interpolator.interpolate(I0, I1, dt)

                    self._write_img(img[0, :260, :346], idx, dest_imgs_dir)
                    timestamps_list.append(t)
                    idx += 1
                    break

        self._write_timestamps(timestamps_list, dest_timestamps_filepath)





    # def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
    #     os.makedirs(dest_imgs_dir, exist_ok=True)
    #     timestamps_list = []
    #     idx = 0
    #     reference_ts = self.reference_timestamps

    #     for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
    #         I0 = img_pair[0][None]
    #         I1 = img_pair[1][None]
    #         t0, t1 = time_pair

    #         # Grayscale → RGB 변환
    #         if I0.shape[-1] == 1:
    #             I0 = np.repeat(I0, 3, axis=-1)
    #         if I1.shape[-1] == 1:
    #             I1 = np.repeat(I1, 3, axis=-1)

    #         # 이 프레임 쌍에 해당하는 보간 타임스탬프만 추출
    #         segment_ts = [t for t in reference_ts if t0 < t < t1]

    #         # 첫 번째 프레임 저장
    #         timestamps_list.append(t0)
    #         self._write_img(I0[0, :260, :346], idx, dest_imgs_dir)
    #         idx += 1

    #         for t in segment_ts:
    #             dt = np.array([(t - t0) / (t1 - t0)], dtype=np.float32)
    #             img, _, _ = self.interpolator.interpolate(I0, I1, dt)
    #             timestamps_list.append(t)
    #             self._write_img(img[0, :260, :346], idx, dest_imgs_dir)
    #             idx += 1

    #     # 마지막 프레임 저장
    #     timestamps_list.append(t1)
    #     self._write_img(I1[0, :260, :346], idx, dest_imgs_dir)

    #     self._write_timestamps(timestamps_list, dest_timestamps_filepath)


    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        # breakpoint()
        if num_bisections == 0:
            return [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            num_bisections = int(np.ceil(np.log(max([flow_mag_0_1_max, flow_mag_1_0_max]))/np.log(2)))

            if num_bisections == 0:
                return [image[0]], [(t0 + t1) / 2]

        left_images, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)
        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        return images, timestamps

    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
        # Copy directory structure.
        def ignore_files(directory, files):
            return [f for f in files if os.path.isfile(os.path.join(directory, f))]
        shutil.copytree(src_dir, dest_dir, ignore=ignore_files, dirs_exist_ok=True)

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = np.clip(img * 255, 0, 255).astype("uint8")
        path = os.path.join(imgs_dir, "%08d.png" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])