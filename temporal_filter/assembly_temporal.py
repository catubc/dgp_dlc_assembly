from moviepy.editor import VideoFileClip
import numpy as np
from pathlib import Path
from skimage.util import img_as_ubyte
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tqdm import tqdm
import yaml
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from util import create_annotated_movie, get_train_config
from deeplabcut.pose_estimation_tensorflow.nnet.pose_net import prediction_layer
from deeplabcut.utils import auxiliaryfunctions

vers = tf.__version__.split('.')
if int(vers[0]) == 1 and int(vers[1]) > 12:
    TF = tf.compat.v1
else:
    TF = tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# %% util functions
def calculate_peaks(numparts, heatmap_avg):
    # Right now there is a score for every part since some parts are likely to need lower thresholds.
    score = np.ones((numparts,)) * 0.000001
    all_peaks = []
    peak_counter = 0
    if len(score) < numparts:
        score = score[:numparts]
        ##logger.ERROR('Not enough scores provided for number of parts')
        # return
    # threshold_detection = params['thre1']
    # tic_localmax=time.time()
    for part in range(numparts):
        map_ori = heatmap_avg[:, :, part]
        map = map_ori
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        peaks_binary = np.logical_and(np.logical_and(np.logical_and(map >= map_left, map >= map_right),
                                                     np.logical_and(map >= map_up, map >= map_down)), map > score[part])
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score_and_id = [x + (map_ori[x[1], x[0]], i + peak_counter,) for i, x in
                                   enumerate(peaks)]  # if x[0]>0 and x[1]>0 ]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    return all_peaks


def zero2nan(x):
    x1 = np.copy(x)
    x1[np.sum(x, 1) == 0, :] = np.nan
    return x1


def temporal_filter_first(markers, na, adds):
    # filter the traces for the first time
    markers_temp = np.copy(markers)
    markers_filter = np.copy(markers_temp)
    end_frame = markers_temp.shape[0]
    wt_bound = 20  # upper bound of the distance that a marker can move between two frames
    cl_bound = 0.7  # lower bound of the confidence level
    part_id = 0
    for animal_id in range(na + adds):  #
        marker_previous = markers_temp[0, animal_id, 0, :]
        for frame_id in range(1, end_frame):
            cl = markers_cen[frame_id, :, :, 0]  # confidence level

            marker_current = markers_temp[frame_id, :, part_id, :]
            marker_current = zero2nan(marker_current)  # set zeros to nans

            marker_previous1 = marker_previous.reshape(1, -1)
            marker_previous1 = zero2nan(marker_previous1)[0]  # set zeros to nans

            move_marker = []
            for marker_i in marker_current:
                move_marker_i = np.sqrt(np.sum((marker_i[1:] - marker_previous1[1:]) ** 2))
                move_marker += [np.sum(move_marker_i)]
            move_marker = np.array(move_marker)
            move_marker[np.isnan(move_marker)] = 1e10  # if the movement is nan, set it to be 1e10
            move_marker_valid_ind = np.where(move_marker <= wt_bound)[0]  # pick the ids whose movement is below wt_bound
            if len(move_marker_valid_ind) == 0:  # if all movements are not valid
                marker_i = marker_previous1
            else:
                cl_valid = cl[move_marker_valid_ind]
                cl_valid_ind = np.where(cl_valid > cl_bound)[0]
                if len(cl_valid_ind) == 0:  # if all of the confidence levels are below 0.7, pick the id with the highest confidence level
                    selected_id = np.argmax(cl_valid)
                    selected_id = move_marker_valid_ind[selected_id]
                else:  # for all confidence levels which are above 0.7, pick the id with the smallest movement
                    move_marker_valid_cl = move_marker_valid_ind[cl_valid_ind]
                    move_marker_valid_cl_ind = move_marker[move_marker_valid_cl]
                    selected_id = np.argmin(move_marker_valid_cl_ind)
                    selected_id = move_marker_valid_cl[selected_id]

                marker_i = marker_current[selected_id, :]
                markers_temp[frame_id, selected_id, part_id, :] = np.nan  # set the identity to be nan if it's already selected

            marker_previous = np.copy(marker_i)
            markers_filter[frame_id, animal_id, part_id, :] = marker_i

    return markers_filter


def temporal_filter_second(markers, na, adds):
    # filter the traces for the second time
    markers_filter = np.copy(markers)
    end_frame = markers_filter.shape[0]
    wt_bound = 20
    cl_bound = 0.7  # lower bound of the confidence level
    part_id = 0
    marker_previous = markers_filter[0, :, part_id, :]
    candidate_id = -np.ones((na + adds, na + adds))
    candidate_id[:, 0] = np.linspace(0, na + adds - 1, na + adds).astype(np.int32)
    candidate_move = candidate_id * 0
    for frame_id in range(1, end_frame):
        cl = markers[frame_id, :, :, 0]
        candidate_id_new = candidate_id.copy()
        for animal_id in range(na + adds):  # [0,1]:#
            marker_current = markers[frame_id, :, part_id, :]
            marker_current = zero2nan(marker_current)  # set zeros to nans

            marker_previous1 = marker_previous[animal_id, :].reshape(1, -1)
            marker_previous1 = zero2nan(marker_previous1)[0]  # set zeros to nans

            move_marker = []
            for marker_i in marker_current:
                move_marker_i = np.sqrt(np.sum((marker_i[1:] - marker_previous1[1:]) ** 2))
                move_marker += [np.sum(move_marker_i)]
            move_marker = np.array(move_marker)
            move_marker[np.isnan(move_marker)] = 1e10  # if the movement is nan, set it to be 1e10
            move_marker_valid_ind = np.where(move_marker <= wt_bound)[0]  # pick the ids whose movement is below wt_bound
            if len(move_marker_valid_ind) > 0:  # if at least one of the movements is valid
                cl_valid = cl[move_marker_valid_ind]
                cl_valid_ind = np.where(cl_valid > cl_bound)[0]
                if len(cl_valid_ind) == 0:  # if all of the confidence levels are below 0.7, pick the id with the highest confidence level
                    selected_id_sortlist = np.argsort(-cl_valid)
                    selected_id_sortlist = move_marker_valid_ind[selected_id_sortlist]
                else:
                    move_marker_valid_cl = move_marker_valid_ind[cl_valid_ind]
                    move_marker_valid_cl_ind = move_marker[move_marker_valid_cl]
                    selected_id_sortlist = np.argsort(move_marker_valid_cl_ind)
                    selected_id_sortlist = move_marker_valid_cl[selected_id_sortlist]

                candidate_id[animal_id, :len(selected_id_sortlist)] = selected_id_sortlist.reshape(-1, )
                candidate_id[animal_id, len(selected_id_sortlist):] = -1

            # based on the candidate id list, make a candidate movement list
            candidate_move_list = move_marker * 0 + 1e10
            for can_j, can_move in enumerate(candidate_id[animal_id, :]):
                if can_move > -1:
                    candidate_move_list[can_j] = move_marker[int(can_move)]
            candidate_move[animal_id, :] = candidate_move_list

        # based on the movement from time t-1 to time t for each id, we can assign ids at t-1 to ids at t
        # we need to make sure: 1. each id at t-1 is only assigned to id at t once, making sure no two ids share the same id at t-1
        #                       2. when two ids at t compete for one id at t-1, the one with the smallest movement will get the assignment
        assigned_list = []
        unique_list = []
        # go over each column from left to right, left has the highest priority
        for c in range(candidate_id.shape[1]):
            candidate_column = candidate_id[:, c]

            # count the occurrence of each id in column c
            uniques, counts = np.unique(candidate_column, return_counts=True)

            # throw away -1
            u1 = np.where(uniques > -1)[0]
            uniques = uniques[u1]
            counts = counts[u1]

            # get the ids who only appear once
            unique_one = uniques[np.where(counts == 1)[0]]
            for uo in unique_one:
                if uo not in unique_list:
                    candidate_uo = np.where(candidate_column == uo)[0][0]
                    if candidate_uo not in assigned_list:
                        candidate_id_new[candidate_uo, :] = uo
                        assigned_list += [candidate_uo]

            # get the ids who appear twice or more
            unique_tm = uniques[np.where(counts > 1)[0]]
            for utm in unique_tm:
                if utm not in unique_list:
                    candidate_utm = np.where(candidate_column == utm)[0]
                    candidate_utm = list(candidate_utm)
                    candidate_move_c = candidate_move[:, c].copy()  # get the candidate movement in column c
                    candidate_move_c[assigned_list] = 1e10  # if one id is already assigned, set the move to be 1e10
                    candidate_move_utm = candidate_move_c[candidate_utm]
                    min_ind = np.argmin(candidate_move_utm)
                    candiate_utm_min = candidate_utm[min_ind]
                    if candiate_utm_min not in assigned_list:
                        candidate_id_new[candiate_utm_min, 0] = utm
                        assigned_list += [candiate_utm_min]

            unique_list += list(uniques)

        full_list = list(range(na + adds))
        not_assigned_list = [v for v in full_list if v not in assigned_list]

        assigned_id = candidate_id_new[assigned_list, 0]
        not_assigned_id = [v for v in full_list if v not in assigned_id]
        candidate_id_new[not_assigned_list, 0] = not_assigned_id
        candidate_id_new[:, 1:] = -1
        candidate_id = candidate_id_new

        marker_previous = marker_current[np.array(candidate_id[:, 0]).astype(np.int32), :]
        markers_filter[frame_id, :, part_id, :] = marker_previous

    return markers_filter


# %%
# snapshot from DLC
snapshot = 'snapshot-100000'
# project path
dlcpath = '/Users/yzhao301/GoogleDriveAnqi/CU_Research/deepgraphpose_kelly/data/mice4/model_data/mice4-cat-2020-07-15'
# the path to the test video
test_video = '/Users/yzhao301/GoogleDriveAnqi/CU_Research/deepgraphpose_kelly/data/mice4/model_data/mice4-cat-2020-07-15/videos/dgp/mice4_test1.mp4'

shuffle = 1
dlc_base_path = Path(dlcpath)
config_path = dlc_base_path / 'config.yaml'
print('config_path', config_path)
cfg = auxiliaryfunctions.read_config(config_path)
trainingsetindex = 0
modelfoldername = auxiliaryfunctions.GetModelFolder(
    cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)

train_path = dlc_base_path / modelfoldername / 'train'
init_weights = str(train_path / snapshot)

# structure info
bodyparts = cfg['multianimalbodyparts']
skeleton = cfg['skeleton']
if skeleton is None:
    skeleton = []
S0 = np.zeros((len(skeleton), len(bodyparts)))
for s in range(len(skeleton)):
    sk = skeleton[s]
    ski = bodyparts.index(sk[0])
    skj = bodyparts.index(sk[1])
    S0[s, ski] = 1
    S0[s, skj] = -1

na = 4  # number of animals
nl = len(skeleton)
nj = len(bodyparts)

# %%
video_clip = VideoFileClip(test_video)
ny_in, nx_in = video_clip.size
n_frames = np.ceil(video_clip.fps * video_clip.duration).astype('int')
print('done')

# load dlc project config file
print('loading dlc project config...', end='')
with open(config_path, 'r') as stream:
    proj_config = yaml.safe_load(stream)
proj_config['video_path'] = None
dlc_cfg = get_train_config(proj_config, shuffle=1)
dlc_cfg.init_weights = init_weights
dlc_cfg.net_type = 'resnet_50'
print(dlc_cfg)
print('dlc_cfg.init_weights', dlc_cfg.init_weights)

# %%
# -------------------
# define model
# -------------------
# sess, net, inputs = initialize_resnet(dlc_cfg, nx_in, ny_in)
TF.reset_default_graph()
inputs = TF.placeholder(tf.float32, shape=[None, None, None, 3])
pn = pose_net(dlc_cfg)

net, end_points = pn.extract_features(inputs)
scope = "pose"
reuse = None
heads = {}
with tf.variable_scope(scope, reuse=reuse):
    heads["part_pred"] = prediction_layer(
        dlc_cfg, net, "part_pred", nj
    )
    heads["locref"] = prediction_layer(
        dlc_cfg, net, "locref_pred", nj * 2
    )

    heads["pairwise_pred"] = prediction_layer(
        dlc_cfg, net, "pairwise_pred", nl * 2
    )
pred = heads['part_pred']

# restore from snapshot
variables_to_restore = slim.get_variables_to_restore(include=["resnet_v1"])
variables_to_restore1 = slim.get_variables_to_restore(include=["pose/part_pred"])
restorer = TF.train.Saver(variables_to_restore + variables_to_restore1)

# initialize tf session
config_TF = TF.ConfigProto()
config_TF.gpu_options.allow_growth = True
sess = TF.Session(config=config_TF)

# initialize weights
sess.run(TF.global_variables_initializer())
sess.run(TF.local_variables_initializer())

# restore resnet from dlc trained weights
restorer.restore(sess, dlc_cfg.init_weights)

# %% collect traces for test video
# -------------------
# extract pose
# -------------------
print('\n')
pbar = tqdm(total=n_frames, desc='processing video frames')
adds = 10  # additional markers to be identified
markers = np.zeros((n_frames, na + adds, nj, 3))
centroid = 6  # pick the 7th marker as the centroid
for i in range(n_frames):

    # scoop out the rock in each frame
    frame = video_clip.get_frame(i / video_clip.fps)
    ff = img_as_ubyte(frame)
    ws1 = 60
    ws2 = 40
    patch = [132, 1080]
    ff_mask = np.zeros((ff.shape))
    ff_mask[(patch[0] - ws1):(patch[0] + ws1), :, :] = ff_mask[(patch[0] - ws1):(patch[0] + ws1), :, :] + 1
    ff_mask[:, (patch[1] - ws2):(patch[1] + ws2), :] = ff_mask[:, (patch[1] - ws2):(patch[1] + ws2), :] + 1
    ff_mask = -np.sign(ff_mask - 2)
    ff = ff * ff_mask

    # feed to the network
    feed_dict = {inputs: ff[None, :, :, :]}
    [pred_s] = sess.run([pred], feed_dict=feed_dict)
    nx_out = pred_s.shape[1]
    ny_out = pred_s.shape[2]
    sig_pred = 1 / (np.exp(-pred_s) + 1)  # confidence map

    # find peaks from the confidence map
    all_peaks = calculate_peaks(nj, sig_pred[0, :, :, :])
    for ind, peak_i in enumerate(all_peaks):
        if len(peak_i) > 0 and ind == centroid:
            peak_i = np.array(peak_i)
            peak_i1 = np.argsort(peak_i[:, 2])
            peak_i = peak_i[peak_i1, :]
            len_peak = len(peak_i)
            if len_peak < na + adds:
                peak_i = np.vstack((np.zeros((na + adds - len_peak, 4)), peak_i))
            peak_i = peak_i[(-na - adds):, :]
            peak_i = peak_i[:, :3]
            peak_i = np.flip(peak_i, 0)
            markers[i, :, ind, :] = peak_i

    pbar.update()

markers = np.flip(markers, 3)

# %%
# filter the traces for the first time
markers_cen = markers[:, :, centroid, None, :]
markers_filter = temporal_filter_first(markers_cen, na, adds)

# %% find the top na traces with the highest confidence levels
top_n = na
markers_filter_cl = np.mean(markers_filter, 0)
markers_filter_cl = np.squeeze(markers_filter_cl[:, :, 0])  # confidence level for each trace
argsort_ind = np.argsort(-markers_filter_cl)  # ind from high confidence level to low
argsort_ind = argsort_ind[:top_n]
markers_filter_cl1 = markers_filter_cl[argsort_ind]
markers_filter1 = markers_filter[:, argsort_ind, :, :]

# %%
# filter the traces for the second time. This is mainly for finetune the filtering to prevent swapping
markers_filter1[:, :, :, 0] = 1
markers_filter2 = temporal_filter_second(markers_filter1, na, 0)

# %% make movie with traces
markers_joint = np.reshape(markers_filter2[:, :, :, 1:], [markers_filter2.shape[0], -1, 2]) * 8 + 4

# -------------------
# save labels
# -------------------
labels = {
    'x': markers_joint[:, :, 1].T,
    'y': markers_joint[:, :, 0].T}
# %
video_clip = VideoFileClip(str(test_video))

# make movie
create_annotated_movie(
    video_clip,
    labels['x'],
    labels['y'],
    filename=test_video[:-4] + '_label.mp4', dotsize=15, colormap='jet')

video_clip.close()















