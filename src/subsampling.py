import numpy as np
import argparse
import os
import pandas as pd
import soundfile as sf
from maad import sound, util
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import datetime as dt
from pathlib import Path

# set python path to correctly use batdetect2 submodule
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src/models/bat_call_detector/batdetect2/"))

from cfg import get_config
from utils.utils import gen_empty_df
from pipeline import pipeline
import models.bat_call_detector.feed_buzz_helper as fbh


def subsample_withpaths(segmented_file_paths, cfg, cycle_length, percent_on):
    necessary_paths = []

    for path in segmented_file_paths:
        if (path['offset'] % cycle_length == 0 # Check if starting position is within recording period; won't need to check rest of boolean if it is
            or ((path['offset']+cfg['segment_duration'])%cycle_length > 0 and (path['offset']+cfg['segment_duration'])%cycle_length <= int(cycle_length*percent_on))):
            necessary_paths.append(path)

    return necessary_paths

def plt_msds_fromdf(location, filename, df, audio_sec, fs, offset, reftimes, times, cycle_length, p_on):
    ## Strip the datetime for year, month, date, and hour from filename
    file_dt = dt.datetime.strptime(f'{filename[:11]}{int(offset/60)%60}{int(offset%60)}', '%Y%m%d_%H%M%S')

    ## Only find numPoints amount of labels from all available seconds
    numPoints = 11
    seconds = np.arange(fs*times[0], fs*times[1]+1)/fs
    idx = np.round(np.linspace(0, len(seconds)-1, numPoints)).astype('int32')
    sec_labels = reftimes[0] + seconds[idx]

    ## Calculate Time Labels for X-Axis using Datetime objects as Strings
    time_labels = np.array([dt.datetime(year=file_dt.year, month=file_dt.month, 
                                        day=file_dt.day, hour=file_dt.hour + int((file_dt.minute + (sec/60))/60), 
                                        minute=(file_dt.minute + int((file_dt.second + sec)/60))%60, second=int((file_dt.second + sec)%60), 
                                        microsecond=np.round(1e6*((file_dt.second + sec)%60-int((file_dt.second + sec)%60))).astype('int32')).strftime('%T.%f')[:-4] 
                                        for sec in sec_labels])

    ## Find x-axis tick locations from all available seconds and convert to samples
    s_ticks = seconds[idx]-times[0]
    s_ticks

    ## Calculate detection parameters from msds output to use for drawing rectangles
    xs_inds, xs_freqs, x_durations, x_bandwidths, det_labels = get_msds_params_from_df(df, reftimes[0]+times)

    ## Create figure
    plt.figure(figsize=(12, 3))
    plt.title(f"{file_dt.date()} in {location} | {cycle_length//60}-min, {100*p_on:.1f}% Duty Cycle")

    ## Plotting Spectrogram with MSDS outputs overlayed
    plt.specgram(audio_sec, Fs=fs, cmap='ocean')
    plt.xlim((0, s_ticks[-1]))
    plt.xlabel("UTC Time (HH:MM:SS)")
    plt.ylabel("Frequency (kHz)")
    plt.xticks(ticks=s_ticks, labels=time_labels)
    # Find y-axis tick locations from specgram-calculated locations and keep limit just in case
    f_ticks = plt.yticks()[0]
    f_ticks = f_ticks[f_ticks <= fs/2]
    plt.yticks(ticks=f_ticks, labels=(f_ticks/1000).astype('int16'))
    ax = plt.gca()
    for i in range(len(xs_inds)):
        rect = Rectangle((xs_inds[i], xs_freqs[i]), 
                        x_durations[i], x_bandwidths[i], 
                        linewidth=1, edgecolor='y', facecolor='none')
        if (np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32') < len(audio_sec) and audio_sec[np.floor((xs_inds[i]+x_durations[i])*fs).astype('int32')] != 0):
            ax.add_patch(rect)
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()

def get_msds_params_from_df(dets:pd.DataFrame, times):
    df = dets
    s_times = df['start_time']
    e_times = df['end_time']
    s_freqs = df['low_freq']
    e_freqs = df['high_freq']
    det_labels = df['event'].values
    xs_inds = s_times[np.logical_and(s_times > times[0], e_times < times[1])].values - times[0]
    xe_inds = e_times[np.logical_and(s_times > times[0], e_times < times[1])].values - times[0]
    xs_freqs = s_freqs[np.logical_and(s_times > times[0], e_times < times[1])].values
    xe_freqs = e_freqs[np.logical_and(s_times > times[0], e_times < times[1])].values
    x_durations = xe_inds - xs_inds
    x_bandwidths = xe_freqs - xs_freqs

    return xs_inds, xs_freqs, x_durations, x_bandwidths, det_labels

def generate_segments(audio_file: Path, output_dir: Path, start_time: float, duration: float):
    """
    Segments audio_file into clips of duration length and saves them to output_dir.
    start_time: seconds
    duration: seconds
    """

    ip_audio = sf.SoundFile(audio_file)

    sampling_rate = ip_audio.samplerate
    # Convert to sampled units
    ip_start = int(start_time * sampling_rate)
    ip_duration = int(duration * sampling_rate)
    ip_end = ip_audio.frames

    output_files = []

    # for the length of the duration, process the audio into duration length clips
    for sub_start in range(ip_start, ip_end, ip_duration):
        sub_end = np.minimum(sub_start + ip_duration, ip_end)

        sub_length = sub_end - sub_start
        ip_audio.seek(sub_start)
        op_audio = ip_audio.read(sub_length)

        # For file names, convert back to seconds 
        op_file = os.path.basename(audio_file.name).replace(" ", "_")
        start_seconds =  sub_start / sampling_rate
        end_seconds =  sub_end / sampling_rate
        op_file_en = "__{:.2f}".format(start_seconds) + "_" + "{:.2f}".format(end_seconds)
        op_file = op_file[:-4] + op_file_en + ".wav"
        
        op_path = os.path.join(output_dir, op_file)
        output_files.append({
            "audio_file": op_path, 
            "offset":  start_time + (sub_start/sampling_rate),
        })
        
        sf.write(op_path, op_audio, sampling_rate, subtype='PCM_16') 

    return output_files 


def get_params(output_dir, tmp_dir, num_processes, segment_duration):
    cfg = get_config()
    cfg["output_dir"] = Path(output_dir)
    cfg["tmp_dir"] = Path(tmp_dir)
    cfg["num_processes"] = num_processes
    cfg['segment_duration'] = segment_duration

    return cfg

def generate_segmented_paths(summer_audio_files, cfg):
    segmented_file_paths = []
    for audio_file in summer_audio_files:
        segmented_file_paths += generate_segments(
            audio_file = audio_file, 
            output_dir = cfg['tmp_dir'],
            start_time = cfg['start_time'],
            duration   = cfg['segment_duration'],
        )
    return segmented_file_paths


## Create necessary mappings from audio to model to file path
def initialize_mappings(necessary_paths, cfg):
    l_for_mapping = [{
        'audio_seg': audio_seg, 
        'model': cfg['models'][0],
        'original_file_name': f"{Path(audio_seg['audio_file']).name[:15]}.WAV",
        } for audio_seg in necessary_paths]

    return l_for_mapping

## Run models and get detections!
def run_models(file_mappings, cfg, csv_name):
    bd_dets = pd.DataFrame()
    for i in tqdm(range(len(file_mappings))):
        cur_seg = file_mappings[i]
        bd_annotations_df = cur_seg['model']._run_batdetect(cur_seg['audio_seg']['audio_file'])
        bd_preds = pipeline._correct_annotation_offsets(
                bd_annotations_df,
                cur_seg['original_file_name'],
                cur_seg['audio_seg']['offset']
            )
        bd_dets = pd.concat([bd_dets, bd_preds])

    bd_dets.to_csv(f"{cfg['output_dir']}/{csv_name}", index=False)

    return bd_dets

def run_subsampling_pipeline(input_dir, cycle_length, percent_on, csv_name, output_dir, tmp_dir):
    cfg = get_params(output_dir, tmp_dir, 4, 30.0)
    summer_audio_files = sorted(list(Path(input_dir).iterdir()))
    segmented_file_paths = generate_segmented_paths(summer_audio_files, cfg)
    print(len(segmented_file_paths), segmented_file_paths)
    ## Get file paths specific to our subsampling parameters
    if (percent_on < 1.0):
        necessary_paths = subsample_withpaths(segmented_file_paths, cfg, cycle_length, percent_on)
    else:
        necessary_paths = segmented_file_paths

    file_path_mappings = initialize_mappings(necessary_paths, cfg)
    print(len(file_path_mappings), file_path_mappings)
    bd_dets = run_models(file_path_mappings, cfg, csv_name)

    return bd_dets

def parse_args():
    """
    Defines the command line interface for the pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="the directory of WAV files to process",
    )
    parser.add_argument(
        "csv_filename",
        type=str,
        help="the file name of the .csv file",
        default="output.csv",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="the directory where the .csv file goes",
        default="output_dir",
    )
    parser.add_argument(
        "temp_dir",
        type=str,
        help="the temp directory where the audio segments go",
        default="output/tmp",
    )
    parser.add_argument(
        "cycle_length",
        type=int,
        help="the desired cycle length in seconds for subsampling",
        default=30,
    )
    parser.add_argument(
        "percent_on",
        type=float,
        help="the desired cycle length in seconds for subsampling",
        default=1/6,
    )

    return vars(parser.parse_args())

if __name__ == "__main__":
    args = parse_args()

    run_subsampling_pipeline(args['input_dir'], args['cycle_length'], args['percent_on'], args['csv_filename'], args['output_dir'], args['temp_dir'])