"""Load data, processes it, save it."""

import sys

sys.path.append("poyo/")
import os
import argparse
import datetime
import logging

import numpy as np
from pynwb import NWBHDF5IO
from scipy.ndimage import binary_dilation, binary_erosion
from einops import rearrange, repeat


from sklearn.preprocessing import StandardScaler

from poyo.data import Data, IrregularTimeSeries, Interval, DatasetBuilder
from poyo.data.dandi_utils import extract_spikes_from_nwbfile, extract_subject_from_nwb
from poyo.utils import find_files_by_extension
from poyo.taxonomy import RecordingTech, Task

logging.basicConfig(level=logging.INFO)


def extract_trials(nwbfile, task):
    r"""Extract trial information from the NWB file. Trials that are flagged as
    "to discard" or where the monkey failed are marked as invalid."""
    trial_table = nwbfile.trials.to_dataframe()

    # rename start and end time columns
    trial_table = trial_table.rename(
        columns={
            "start_time": "start",
            "stop_time": "end",
        }
    )

    # GIANNIS: there's an error with the 'split' column, dropping
    trial_table = trial_table.drop(columns=["split"])
    trials = Interval.from_dataframe(trial_table)

    # GIANNIS: task is the same, not doing this
    # if task == "center_out_reaching":
    #     trials.is_valid = np.logical_and(
    #         np.logical_and(trials.result == "R", ~(np.isnan(trials.target_id))),
    #         (trials.end - trials.start) < 6.0,
    #     )

    # elif task == "random_target_reaching":
    #     trials.is_valid = np.logical_and(
    #         np.logical_and(trials.result == "R", trials.num_attempted == 4),
    #         (trials.end - trials.start) < 10.0,
    #     )

    # GIANNIS: all valid afaik
    trials.is_valid = np.ones(len(trials), dtype=bool)

    return trials


def plot_histograms(data, output_dir, vel_labels=["x", "y"]):
    """
    Plots histograms of the input numpy array and saves them to disk.

    Parameters:
    - data: numpy array, shape (n_samples,) or (n_samples, n_features)
    - output_dir: str, directory to save the histograms
    """
    import matplotlib.pyplot as plt
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if data.ndim == 1:
        data = data[:, None]
    for i in range(data.shape[1]):
        plt.figure()
        non_zero_bucket_flags = np.abs(data[:, i]) > 10
        plt.hist(
            data[:, i][non_zero_bucket_flags],
            bins=300,
            alpha=0.5,
            color="blue",
            edgecolor="black",
        )
        plt.title(f"Histogram for Velocity {vel_labels[i]}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"rtt_histogram_vel_{vel_labels[i]}.png"))
        plt.close()


def extract_behavior(nwbfile, trials, task):
    """Extract behavior from the NWB file.

    ..note::
        Cursor position and target position are in the same frame of reference.
        They are both of size (sequence_len, 2). Finger position can be either 3d or 6d,
        depending on the sequence. # todo investigate more
    """

    # GIANNIS: This is *NOT* the cursor velocity, it's the finger velocity.
    # I'm just trying not to break downstream processing.
    finger_vel_all = nwbfile.processing["behavior"]["finger_vel"]
    cursor_vel = finger_vel_all.data[:]

    # TODO: this better be correct, no explicit timestamps available in the data...
    timestamps = finger_vel_all.starting_time + finger_vel_all.conversion * np.arange(
        len(cursor_vel)
    )

    ## SUBSAMPLING
    timestamps = timestamps[::10]
    cursor_vel = cursor_vel[::10] * 2.5  # RTT SPECIFIC - SCALING FOR SAME DATA RANGES

    # normalization

    # cursor_vel = cursor_vel / 20.0

    # scaler = StandardScaler()#l
    # cursor_vel = scaler.fit_transform(cursor_vel)#l

    # create a behavior type segmentation mask
    subtask_index = np.ones_like(timestamps, dtype=np.int64) * int(Task.REACHING.RANDOM)
    if task == "center_out_reaching":
        for i in range(len(trials)):
            # first we check whether the trials are valid or not
            if trials.is_valid[i]:
                subtask_index[
                    (timestamps >= trials.target_on_time[i])
                    & (timestamps < trials.go_cue_time[i])
                ] = int(Task.REACHING.HOLD)
                subtask_index[
                    (timestamps >= trials.go_cue_time[i]) & (timestamps < trials.end[i])
                ] = int(Task.REACHING.REACH)
                subtask_index[
                    (timestamps >= trials.start[i])
                    & (timestamps < trials.target_on_time[i])
                ] = int(Task.REACHING.RETURN)
    elif (
        task == "random_target_reaching"
    ):  # GIANNIS: this is the default task class - all trials are valid
        for i in range(len(trials)):
            if trials.is_valid[i]:
                pass  # no go_cue_time available here...
                # subtask_index[
                #     (timestamps >= trials.start[i])
                #     & (timestamps < trials.go_cue_time[i])  # GIANNIS: changed slightly
                # ] = int(Task.REACHING.HOLD)

    # sometimes monkeys get angry, we want to identify the segments where the hand is
    # moving too fast, and mark them as outliers
    # we use the norm of the acceleration to identify outliers
    # hand_acc_norm = np.linalg.norm(cursor_acc, axis=1)
    # mask = hand_acc_norm > 1500.0
    # mask = binary_dilation(mask, structure=np.ones(2, dtype=bool))
    # subtask_index[mask] = int(Task.REACHING.OUTLIER)

    # # we also want to identify out of bound segments
    # mask = np.logical_or(cursor_pos[:, 0] < -10, cursor_pos[:, 0] > 10)
    # mask = np.logical_or(mask, cursor_pos[:, 1] < -10)
    # mask = np.logical_or(mask, cursor_pos[:, 1] > 10)
    # # dilate than erode
    # mask = binary_dilation(mask, np.ones(400, dtype=bool))
    # mask = binary_erosion(mask, np.ones(100, dtype=bool))
    # subtask_index[mask] = int(Task.REACHING.OUTLIER)

    # cursor = IrregularTimeSeries(
    #     timestamps=timestamps,
    #     pos=cursor_pos,
    #     vel=cursor_vel,
    #     acc=cursor_acc,
    #     subtask_index=subtask_index,
    #     domain="auto",
    # )

    # GIANNIS: dummy data, cannot use. Also not filtering for any outliers at this point.
    cursor = IrregularTimeSeries(
        timestamps=timestamps,
        pos=cursor_vel,
        vel=cursor_vel,
        acc=cursor_vel,
        subtask_index=subtask_index,
        domain="auto",
    )

    return cursor


# GIANNIS: Why oh whyyy would we need this???
def calc_trial_overlap_mask(trials, preferential_acceptance=None):
    # figure out non-offenders by invalid start and end times
    no_end_violation = (trials.end[:-1] <= trials.start[1:]).astype(bool)  # (N-1,)
    no_start_violation = (trials.start[1:] >= trials.end[:-1]).astype(bool)  # (N-1,)

    # if we have a preferential acceptance, ignore violations on the preferred side
    if preferential_acceptance == "start":
        no_start_violation = np.ones_like(no_end_violation, dtype=bool)  # (N-1,)
    elif preferential_acceptance == "end":
        no_end_violation = np.ones_like(no_start_violation, dtype=bool)  # (N-1,)

    # combine the two to get a mask of trials that do not overlap, rejecting end time violators
    no_overlap = np.ones(len(trials), dtype=bool)  # (N,) - by default all are ok
    no_overlap[1:-1] = np.logical_and(
        no_end_violation[1:], no_start_violation[:-1]
    )  # (N-2,)
    no_overlap[0] = no_end_violation[0]  # (1,)
    no_overlap[-1] = no_start_violation[-1]  # (1,)
    return no_overlap


def main():
    # use argparse to get arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./raw")
    parser.add_argument("--output_dir", type=str, default="./processed")

    args = parser.parse_args()

    assert os.path.isdir(
        args.input_dir
    ), f"Input directory {args.input_dir} does not exist."
    if not os.path.isdir(args.output_dir):
        print(f"Output directory non-existent, creating it at: {args.output_dir}")
        os.makedirs(args.output_dir)

    # intiantiate a DatasetBuilder which provides utilities for processing data
    db = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="mc_rtt",
        origin_version="dandi/000129/sub-Indy",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000129",
        description="Data from monkey Indy performing self-paced random target reaching task. "
        "This file contains continuous segments of the full session on 2017-02-02 that can be "
        "used for training models for the Neural Latents Benchmark.",
    )

    db1 = DatasetBuilder(
        raw_folder_path=args.input_dir,
        processed_folder_path=args.output_dir,
        # metadata for the dataset
        experiment_name="mc_rtt",
        origin_version="dandi/000129/sub-Indy",
        derived_version="1.0.0",
        source="https://dandiarchive.org/dandiset/000129",
        description="Data from monkey Indy performing self-paced random target reaching task. "
        "This file contains continuous segments of the full session on 2017-02-02 that can be "
        "used for training models for the Neural Latents Benchmark.",
    )

    global_vel = []
    global_vel_max_len = 0
    for i, file_path in enumerate(find_files_by_extension(db1.raw_folder_path, ".nwb")):
        if "behavior" not in str(file_path).lower():
            continue
        with db1.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            # extract subject metadata
            # this dataset is from dandi, which has structured subject metadata, so we
            # can use the helper function extract_subject_from_nwb
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")

            sortset_id = f"{subject.id}_{recording_date}"
            task = (
                "center_out_reaching" if "CO" in file_path else "random_target_reaching"
            )
            session_id = f"{sortset_id}_{task}"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            print(f"Registering: {session_id}")

            # extract spiking activity
            # this data is from dandi, we can use our helper function
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            trials = extract_trials(nwbfile, task)

            # extract behavior
            cursor = extract_behavior(nwbfile, trials, task)
            global_vel.append(cursor.vel)
            global_vel_max_len = max(global_vel_max_len, len(cursor.vel))
            io.close()
            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=cursor,
                # domain
                domain=cursor.domain,
            )

            session.register_data(data)
    global_vel = np.concatenate(global_vel, axis=0)
    scaler = StandardScaler()  # l
    global_vel_fill = scaler.fit_transform(global_vel)  # l

    # iterate over the .nwb files and extract the data from each
    for i, file_path in enumerate(find_files_by_extension(db.raw_folder_path, ".nwb")):
        if "behavior" not in str(file_path).lower():
            continue
        logging.info(f"Processing file: {file_path}")

        # each file contains data from one session. a session is unique and has one
        # associated subject and one associated sortset.
        with db.new_session() as session:
            # open file
            io = NWBHDF5IO(file_path, "r")
            nwbfile = io.read()

            # extract subject metadata
            # this dataset is from dandi, which has structured subject metadata, so we
            # can use the helper function extract_subject_from_nwb
            subject = extract_subject_from_nwb(nwbfile)
            session.register_subject(subject)

            # extract experiment metadata
            recording_date = nwbfile.session_start_time.strftime("%Y%m%d")

            sortset_id = f"{subject.id}_{recording_date}"
            task = (
                "center_out_reaching" if "CO" in file_path else "random_target_reaching"
            )
            session_id = f"{sortset_id}_{task}"

            # register session
            session.register_session(
                id=session_id,
                recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
                task=Task.REACHING,
            )

            # extract spiking activity
            # this data is from dandi, we can use our helper function
            spikes, units = extract_spikes_from_nwbfile(
                nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
            )

            # register sortset
            session.register_sortset(
                id=sortset_id,
                units=units,
            )

            # extract data about trial structure
            trials = extract_trials(nwbfile, task)

            # extract behavior
            cursor = extract_behavior(nwbfile, trials, task)

            cursor.vel = scaler.transform(cursor.vel)

            # close file
            io.close()

            # register session
            data = Data(
                # neural activity
                spikes=spikes,
                units=units,
                # stimuli and behavior
                trials=trials,
                cursor=cursor,
                # domain
                domain=cursor.domain,
            )

            session.register_data(data)

            # GIANNIS: we need to manually sort through overlaps and discard some trials...
            # with "start" policy, keeping 846 out of 1080
            mask_disjoint = calc_trial_overlap_mask(
                trials, preferential_acceptance="start"
            )

            # split trials into train, validation and test
            successful_trials = trials.select_by_mask(
                np.logical_and(trials.is_valid, mask_disjoint)
            )

            # successful_trials = trials.select_by_mask(trials.is_valid)

            # GIANNIS: train trials were never recovered or used, reinstated this
            train_trials, valid_trials, test_trials = successful_trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=96
            )

            # # GIANNIS: dilation needs to be manually tuned...
            # train_sampling_intervals = data.domain.difference(
            #     (valid_trials | test_trials).dilate(1.0)
            # )

            session.register_split("train", train_trials)
            session.register_split("valid", valid_trials)
            session.register_split("test", test_trials)

            # save data to disk
            session.save_to_disk()

    # all sessions added, finish by generating a description file for the entire dataset
    db.finish()


if __name__ == "__main__":
    main()
