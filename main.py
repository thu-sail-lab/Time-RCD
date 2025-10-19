# -*- coding: utf-8 -*-
# Author: Qinghua Liu <liu.11085@osu.edu>
# License: Apache-2.0 License

import pandas as pd
import torch
import random, argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from .evaluation.metrics import get_metrics
from .utils.slidingWindows import find_length_rank
from .model_wrapper import *
from .HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict
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
    Multi = parser.parse_args().mode == 'multi'
    # Initialize list to store all results
    all_results = []
    all_logits = []
    if Multi:
        filter_list = [
                "GHL",
                "Daphnet",
                "Exathlon",
                "Genesis",
                "OPP",
                "SMD",
                # "SWaT",
                # "PSM",
                "SMAP",
                "MSL",
                "CreditCard",
                "GECCO",
                "MITDB",
                "SVDB",
                "LTDB",
                "CATSv2",
                "TAO"
            ]
        base_dir = '/home/lihaoyang/Huawei/TSB-AD/Datasets/TSB-AD-M/'
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

                "IOPS",
                "MGAB",
                "NAB",
                "NEK",
                # "Power",
                # "SED",
                "Stock",
                "TODS",
                "WSD",
                "YAHOO",
                "UCR"
                ]
        base_dir = '/home/lihaoyang/Huawei/TSB-AD/Datasets/TSB-AD-U/'
        files = os.listdir(base_dir)



    # ## ArgumentParser
    for file in files:

        if any(filter_item in file for filter_item in filter_list):
            print(f"Skipping file: {file} due to filter criteria.")
            continue

        # Clear GPU memory before processing each file
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        args = parser.parse_args()
        if Multi:
            Optimal_Det_HP = Optimal_Multi_algo_HP_dict[args.AD_Name]
        else:
            Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
        # try:
            # Read data using a proper path join
        df_path = os.path.join(args.data_direc, args.filename)
        df = pd.read_csv(df_path).dropna()
        data = df.iloc[:, 0:-1].values.astype(float)
        print(f"Processing file: {args.filename}, Data shape: {data.shape}")
        label = df['Label'].astype(int).to_numpy()
        print(f"Label shape: {label.shape}")

        slidingWindow = find_length_rank(data, rank=1)
        train_index = args.filename.split('.')[0].split('_')[-3]
        data_train = data[:int(train_index), :]
        test_data  = data[int(train_index):, :]
        label_test = label[int(train_index):]


        print(f"Train data shape: {data_train.shape}, Test data shape: {test_data.shape}, Label test shape: {label_test.shape}")

        print(f"Optimal Hyperparameters for {args.AD_Name}: {Optimal_Det_HP}")
        logits = None  # ensure defined irrespective of branch

        print(f"Running {args.AD_Name} on {args.filename}...")
        if args.AD_Name in Semisupervise_AD_Pool:
            output = run_Semisupervise_AD(args.AD_Name, data_train, test_data, **Optimal_Det_HP)
        elif args.AD_Name in Unsupervise_AD_Pool:
            if args.AD_Name == 'Time_RCD':
                # For Time_RCD, we need to pass the test data directly
                output, logits = run_Unsupervise_AD(args.AD_Name, data_train, test_data, Multi=Multi, **Optimal_Det_HP)
            else:
                output = run_Unsupervise_AD(args.AD_Name, data_train, test_data, **Optimal_Det_HP)
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

            print(f"Original shapes - Output: {output.shape}, Label: {label_test.shape}")
            print(f"Aligned shapes - Output: {output_aligned.shape}, Label: {label_aligned.shape}")

            evaluation_result = get_metrics(output_aligned, label_aligned, slidingWindow=slidingWindow, pred=output_aligned > (np.mean(output_aligned)+3*np.std(output_aligned)))
            evaluation_result_logits = None
            if logits is not None:
                evaluation_result_logits = get_metrics(logits_aligned, label_aligned, slidingWindow=slidingWindow, pred=logits_aligned > (np.mean(logits_aligned)+3*np.std(logits_aligned)))
            print('Evaluation Result: ', evaluation_result)

            # Prepare result dictionary with filename and all metrics
            result_dict = {
                'filename': args.filename,
                'AD_Name': args.AD_Name,
                'sliding_window': slidingWindow,
                'train_index': train_index,
                'data_shape': f"{data.shape[0]}x{data.shape[1]}",
                'output_length': len(output),
                'label_length': len(label_test),  # Use label_test length
                'aligned_length': min_length,
                **evaluation_result  # Unpack all evaluation metrics
            }
            all_results.append(result_dict)

            print(f"Results for {args.filename}: {result_dict}")
            if logits is not None:
                logit_dict = {
                    'filename': args.filename,
                    'AD_Name': args.AD_Name,
                    'sliding_window': slidingWindow,
                    'train_index': train_index,
                    'data_shape': f"{data.shape[0]}x{data.shape[1]}",
                    'output_length': len(logits),
                    'label_length': len(label_test),  # Use label_test length
                    'aligned_length': min_length,
                    **evaluation_result_logits  # Unpack all evaluation metrics for logits
                }
                all_logits.append(logit_dict)
            print(f"Logits results for {args.filename}: {logit_dict}" if logits is not None else "No logits available")
            # Save value, label, and anomaly scores to pickle file
            if args.save:
                output_filename = f'{args.filename.split(".")[0]}_results.pkl'
                output_path = os.path.join(
                    os.path.join(os.getcwd(), (f"{'Multi' if Multi else 'Uni'}_"+args.AD_Name), output_filename))
                if not os.path.exists(output_path):
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                pd.DataFrame({
                    'value': test_data[:min_length].tolist(),
                    'label': label_aligned.tolist(),
                    'anomaly_score': output_aligned.tolist(),
                    'logits': logits_aligned.tolist() if logits is not None else None
                }).to_pickle(output_path)
                print(f'Results saved to {output_path}')
        else:
            print(f'At {args.filename}: '+output)
            # Save error information as well
            result_dict = {
                'filename': args.filename,
                'AD_Name': args.AD_Name,
                'sliding_window': None,
                'train_index': None,
                'data_shape': None,
                'error_message': output
            }
            all_results.append(result_dict)

    # Convert results to DataFrame and save to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        # win_size =  str(Optimal_Det_HP['win_size']) if Optimal_Det_HP['win_size'] else ""
        output_filename = f'{"Multi" if Multi else "Uni"}_{args.AD_Name}.csv'
        results_df.to_csv(output_filename, index=False)
        print(f"\nAll results saved to {output_filename}")
        print(f"Total file processed: {len(all_results)}")
        print(f"Results shape: {results_df.shape}")
        if all_logits:
            logits_df = pd.DataFrame(all_logits)
            logits_output_filename = f'{"Multi" if Multi else "Uni"}_{args.AD_Name}.csv'
            logits_df.to_csv(logits_output_filename, index=False)
            print(f"Logits results saved to {logits_output_filename}")
    else:
        print("No results to save.")
