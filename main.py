# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import random, argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from evaluation.metrics import get_metrics, get_metrics_optimized
from utils.slidingWindows import find_length_rank
from model_wrapper import *
from HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
import os
# Cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import os
print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())
import pickle


def parse_window_sizes(window_sizes_arg):
    if not window_sizes_arg:
        return []

    parsed = []
    for token in window_sizes_arg.split(','):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid window size '{token}'. Please provide integers separated by commas.") from exc
        if value <= 0:
            raise ValueError(f"Window size must be positive, got {value}.")
        parsed.append(value)

    # Keep order but remove duplicates.
    return list(dict.fromkeys(parsed))


def build_window_test_configs(base_hp, requested_window_sizes):
    if not requested_window_sizes:
        return [(dict(base_hp), None, None)]

    target_window_key = None
    for candidate in ('win_size', 'window_size', 'slidingWindow'):
        if candidate in base_hp:
            target_window_key = candidate
            break

    if target_window_key is None:
        print("Warning: --test_window_sizes was provided, but current model has no supported window hyperparameter. Running default configuration only.")
        return [(dict(base_hp), None, None)]

    configs = []
    for window_size in requested_window_sizes:
        hp = dict(base_hp)
        hp[target_window_key] = window_size
        configs.append((hp, target_window_key, window_size))

    return configs


def build_window_suffix(window_sizes):
    if not window_sizes:
        return ''
    return '_wins-' + '-'.join(str(w) for w in window_sizes)


def build_metric_preview(metric_dict):
    if not isinstance(metric_dict, dict):
        return ""
    preview_keys = [
        'F1',
        'f1',
        'best_f1',
        'Affiliation_F',
        'AUC_ROC',
        'AUC_PR',
    ]
    shown = []
    for key in preview_keys:
        if key in metric_dict:
            value = metric_dict[key]
            if isinstance(value, (int, float, np.floating)):
                shown.append(f"{key}={value:.4f}")
            else:
                shown.append(f"{key}={value}")
        if len(shown) >= 2:
            break
    return ', '.join(shown)


def get_result(filename):
    pickle_filename = filename.replace('.csv', '_results.pkl')
    df = pickle.load(open(pickle_filename, 'rb'))

    return df['anomaly_score'].to_numpy()

if __name__ == '__main__':
    # Resolve dataset directory relative to this file (portable across machines)
    parser = argparse.ArgumentParser(description='Running TSB-AD')
    parser.add_argument('--mode', type=str, default='uni', choices=['uni', 'multi'],
                    help='Encoder mode: uni for univariate, multi for multivariate')
    parser.add_argument('--AD_Name', type=str, default='Time_RCD')
    parser.add_argument('--filename', type=str, default='')
    parser.add_argument('--data_direc', type=str, default='')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--metrics_mode', type=str, default='fast', choices=['default', 'fast'],
                    help='Metric calculation mode. fast uses parallelized metrics.')
    parser.add_argument('--skip_logits_metrics', action='store_true',
                    help='Skip metric calculation for logits to reduce runtime.')
    parser.add_argument('--metrics_heavy_workers', type=int, default=0,
                    help='Workers for heavy metrics in fast mode. 0 means auto.')
    parser.add_argument('--metrics_light_workers', type=int, default=0,
                    help='Workers for light metrics in fast mode. 0 means auto.')
    parser.add_argument('--metrics_f1t_splits', type=int, default=800,
                    help='Number of threshold splits for F1_T in fast mode (lower is faster).')
    parser.add_argument('--metrics_f1t_chunk_size', type=int, default=25,
                    help='Chunk size for F1_T threshold processing in fast mode.')
    parser.add_argument('--test_window_sizes', type=str, default='',
                    help='Comma-separated window sizes for multi-window testing, e.g. "512,1024,2048".')
    args = parser.parse_args()
    Multi = args.mode == 'multi'
    requested_window_sizes = parse_window_sizes(args.test_window_sizes)
    output_window_suffix = build_window_suffix(requested_window_sizes)

    def compute_metrics(score_arr, label_arr, sw, pred_arr):
        if args.metrics_mode == 'fast':
            return get_metrics_optimized(
                score_arr,
                label_arr,
                slidingWindow=sw,
                pred=pred_arr,
                heavy_workers=args.metrics_heavy_workers if args.metrics_heavy_workers > 0 else None,
                light_workers=args.metrics_light_workers if args.metrics_light_workers > 0 else None,
                f1_t_n_splits=max(50, args.metrics_f1t_splits),
                f1_t_chunk_size=max(1, args.metrics_f1t_chunk_size),
            )
        return get_metrics(score_arr, label_arr, slidingWindow=sw, pred=pred_arr)
    # Initialize list to store all results
    all_results = []
    all_logits = []
    if Multi:
        filter_list = [
                # "GHL",
                # "Daphnet",
                # "Exathlon",
                # "Genesis",
                # "OPP",
                # "SMD",
                # "SWaT",
                # "PSM",
                # "SMAP",
                # "MSL",
                # "CreditCard",
                # "GECCO",
                # "MITDB",
                # "SVDB",
                # "LTDB",
                # "CATSv2",
                # "TAO"
            ]
        base_dir = 'datasets/TSB-AD-M/'
        files = os.listdir(base_dir)
    else:
        filter_list = [
                    "Daphnet",
                    "CATSv2",
                    "SWaT",
                    "LTDB",
                    "TAO",
                    "Exathlon",
                    "MITDB",
                    "MSL",
                    "SMAP",
                    "SMD",
                    "SVDB",
                    "OPP",

                # "IOPS",
                # "MGAB",
                # "NAB",
                # "NEK",
                # "Power",
                # "SED",
                # "Stock",
                # "TODS",
                # "WSD",
                # "YAHOO",
                # "UCR"
                ]
        base_dir = 'datasets/TSB-AD-U/'
        files = os.listdir(base_dir)



    target_files = [
        file for file in files
        if not any(filter_item in file for filter_item in filter_list)
    ]
    total_datasets = len(target_files)

    if total_datasets == 0:
        print("No datasets matched after applying filter criteria.")

    # ## ArgumentParser
    for dataset_idx, file in enumerate(target_files, start=1):

        # Clear GPU memory before processing each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Set the file-specific values
        args.filename = file
        args.data_direc = base_dir
        
        if Multi:
            base_optimal_det_hp = dict(Optimal_Multi_algo_HP_dict[args.AD_Name])
        else:
            base_optimal_det_hp = dict(Optimal_Uni_algo_HP_dict[args.AD_Name])
        # try:
            # Read data using a proper path join
        df_path = os.path.join(args.data_direc, args.filename)
        df = pd.read_csv(df_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()

        slidingWindow = find_length_rank(data, rank=1)
        train_index = args.filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]
        test_data  = data[int(train_index):, :]
        label_test = label[int(train_index):]

        hp_test_configs = build_window_test_configs(base_optimal_det_hp, requested_window_sizes)
        total_windows = len(hp_test_configs)
        for window_idx, (det_hp, window_key, tested_window) in enumerate(hp_test_configs, start=1):
            logits = None  # ensure defined irrespective of branch

            datasets_left = total_datasets - dataset_idx

            if tested_window is not None:
                print(
                    f"[{dataset_idx}/{total_datasets}] [{window_idx}/{total_windows}] "
                    f"dataset={args.filename} testing {window_key}={tested_window} | datasets_left={datasets_left}"
                )
            else:
                print(
                    f"[{dataset_idx}/{total_datasets}] [{window_idx}/{total_windows}] "
                    f"dataset={args.filename} testing default_window | datasets_left={datasets_left}"
                )

            if args.AD_Name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(args.AD_Name, data_train, test_data, **det_hp)
            elif args.AD_Name in Unsupervise_AD_Pool:
                if args.AD_Name == 'Time_RCD':
                    # For Time_RCD, we need to pass the test data directly
                    output, logits = run_Unsupervise_AD(args.AD_Name, data_train, test_data, Multi=Multi, **det_hp)
                else:
                    output = run_Unsupervise_AD(args.AD_Name, data_train, test_data, **det_hp)
            else:
                raise Exception(f"{args.AD_Name} is not defined")

            if isinstance(output, np.ndarray):
                # output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()

                # Fix shape mismatch issue - ensure output and labels have the same length
                min_length = min(len(output), len(label_test))  # Use label_test instead of label
                output_aligned = output[:min_length]
                label_aligned = label_test[:min_length]
                logits_aligned = None
                if logits is not None:
                    logits_aligned = logits[:min_length]


                evaluation_result = compute_metrics(
                    output_aligned,
                    label_aligned,
                    slidingWindow,
                    output_aligned > (np.mean(output_aligned)+3*np.std(output_aligned))
                )
                evaluation_result_logits = None
                if logits is not None and not args.skip_logits_metrics:
                    evaluation_result_logits = compute_metrics(
                        logits_aligned,
                        label_aligned,
                        slidingWindow,
                        logits_aligned > (np.mean(logits_aligned)+3*np.std(logits_aligned))
                    )

                metric_preview = build_metric_preview(evaluation_result)
                if metric_preview:
                    print(f"  -> done ({metric_preview})")
                else:
                    print("  -> done")

                # Prepare result dictionary with filename and all metrics
                result_dict = {
                    'filename': args.filename,
                    'AD_Name': args.AD_Name,
                    'sliding_window': slidingWindow,
                    'test_window_key': window_key,
                    'test_window_value': tested_window,
                    'train_index': train_index,
                    'data_shape': f"{data.shape[0]}x{data.shape[1]}",
                    'output_length': len(output),
                    'label_length': len(label_test),  # Use label_test length
                    'aligned_length': min_length,
                    **evaluation_result  # Unpack all evaluation metrics
                }
                all_results.append(result_dict)

                if logits is not None and evaluation_result_logits is not None:
                    logit_dict = {
                        'filename': args.filename,
                        'AD_Name': args.AD_Name,
                        'sliding_window': slidingWindow,
                        'test_window_key': window_key,
                        'test_window_value': tested_window,
                        'train_index': train_index,
                        'data_shape': f"{data.shape[0]}x{data.shape[1]}",
                        'output_length': len(logits),
                        'label_length': len(label_test),  # Use label_test length
                        'aligned_length': min_length,
                        **evaluation_result_logits  # Unpack all evaluation metrics for logits
                    }
                    all_logits.append(logit_dict)
                # Save value, label, and anomaly scores to pickle file
                if args.save:
                    dataset_name = args.filename.split('.')[0]
                    if tested_window is not None:
                        output_filename = f'{dataset_name}_{window_key}_{tested_window}_results.pkl'
                    else:
                        output_filename = f'{dataset_name}_results.pkl'
                    output_path = os.path.join(
                        os.path.join(os.getcwd(), (f"{'Multi' if Multi else 'Uni'}_"+args.AD_Name+"_v4"), output_filename))
                    if not os.path.exists(output_path):
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    pd.DataFrame({
                        'value': test_data[:min_length].tolist(),
                        'label': label_aligned.tolist(),
                        'anomaly_score': output_aligned.tolist(),
                        'logits': logits_aligned.tolist() if logits is not None else None
                    }).to_pickle(output_path)
            else:
                print(f'  -> failed: {output}')
                # Save error information as well
                result_dict = {
                    'filename': args.filename,
                    'AD_Name': args.AD_Name,
                    'sliding_window': None,
                    'test_window_key': window_key,
                    'test_window_value': tested_window,
                    'train_index': None,
                    'data_shape': None,
                    'error_message': output
                }
                all_results.append(result_dict)

    # Convert results to DataFrame and save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        # win_size =  str(Optimal_Det_HP['win_size']) if Optimal_Det_HP['win_size'] else ""
        output_filename = f'{"Multi" if Multi else "Uni"}_{args.AD_Name}_v4{output_window_suffix}.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\nAll results saved to {output_filename}")
        print(f"Total file processed: {len(all_results)}")
        print(f"Results shape: {results_df.shape}")
        if all_logits:
            logits_df = pd.DataFrame(all_logits)
            logits_output_filename = f'{"Multi" if Multi else "Uni"}_{args.AD_Name}_v4{output_window_suffix}_logits.csv'
            logits_df.to_csv(logits_output_filename, index=False)
            print(f"Logits results saved to {logits_output_filename}")
    else:
        print("No results to save.")
