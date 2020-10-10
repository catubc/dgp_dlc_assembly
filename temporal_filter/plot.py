import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import os
import pickle
from skimage.draw import circle
from skimage.util import img_as_ubyte
from tqdm import tqdm
import yaml



def get_clip_frames(clip, frames_idxs, num_channels=3):
    numframes = len(frames_idxs)
    xlim, ylim = clip.size
    fps = clip.fps

    frames_arr = np.zeros((numframes, ylim, xlim, num_channels))

    for idx, frame_idx in enumerate(frames_idxs):
        # print(frame_idx)
        frame_idx_sec = frame_idx / fps
        frames_arr[idx] = clip.get_frame(frame_idx_sec)
    return frames_arr


def make_cmap(number_colors, cmap="cool"):
    color_class = plt.cm.ScalarMappable(cmap=cmap)
    C = color_class.to_rgba(np.linspace(0, 1, number_colors))
    colors = (C[:, :3] * 255).astype(np.uint8)
    return colors


def create_annotated_movie(
        clip, df_x, df_y, mask_array=None, dotsize=5, colormap="cool", filename="movie.mp4"):
    if mask_array is None:
        mask_array = ~np.isnan(df_x)
    # ------------------------------
    # Get number of colorvars to plot

    number_body_parts, T = df_x.shape

    # Set colormap for each color
    colors = make_cmap(number_body_parts, cmap=colormap)

    nx, ny = clip.size
    duration = int(clip.duration - clip.start)
    fps = clip.fps
    nframes = int(duration * fps)

    print(
            "Duration of video [s]: ",
            round(duration, 2),
            ", recorded with ",
            round(fps, 2),
            "fps!",
    )

    # print("Overall # of frames: ", nframes, "with cropped frame dimensions: ", nx, ny)
    # print("Generating frames and creating video.")

    # add marker to each frame t, where t is in sec
    def add_marker(get_frame, t):

        image = get_frame(t * 1.0)

        # frame [ny x ny x 3]
        frame = image.copy()
        # convert from sec to indices
        index = int(np.round(t * 1.0 * fps))

        if index % 1000 == 0:
            print("\nTime frame @ {} [sec] is {}".format(t, index))

        for bpindex in range(number_body_parts):

            if index >= T:
                print('SKipped frame {}, marker {}'.format(index, bpindex))
                continue
            if mask_array[bpindex, index]:
                xc = min(int(df_x[bpindex, index]), nx - 1)
                yc = min(int(df_y[bpindex, index]), ny - 1)
                # rr, cc = circle_perimeter(yc, xc, dotsize, shape=(ny, nx))
                rr, cc = circle(yc, xc, dotsize, shape=(ny, nx))
                frame[rr, cc, :] = colors[bpindex]

        return frame

    clip_marked = clip.fl(add_marker)

    clip_marked.write_videofile(
            str(filename), codec="mpeg4", fps=fps, bitrate="1000k"
    )
    clip_marked.close()
    return
