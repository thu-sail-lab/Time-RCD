import sys
import time
from .basic_metrics import basic_metricor, generate_curve
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
import multiprocessing

import multiprocessing
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import time

# ============== Parallelized Affiliation ==============

def _compute_auc_roc(labels, score):
    grader = basic_metricor()
    try:
        return grader.metric_ROC(labels, score)
    except Exception:
        return 0.0

def _compute_auc_pr(labels, score):
    grader = basic_metricor()
    try:
        return grader.metric_PR(labels, score)
    except Exception:
        return 0.0

def _compute_vus(labels, score, slidingWindow, version):
    try:
        _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version)
        return VUS_ROC, VUS_PR
    except Exception:
        return 0.0, 0.0

def _compute_pointf1(labels, score):
    # print("Evaluating F1 standard...")
    grader = basic_metricor()
    try:
        # print("Using chunked parallel F1 computation...")
        return grader.metric_standard_F1_chunked(
            true_labels=labels, 
            anomaly_scores=score,
            chunk_size=25,  # Process 25 thresholds per chunk
            num_workers=4   # Use 4 parallel workers
        )
    except Exception:
        # print("F1 standard computation failed, returning zeros.")
        return {'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0}

def _compute_pointf1pa(labels, score):
    grader = basic_metricor()
    try:
        return grader.metric_PointF1PA_chunked(
            label=labels, 
            score=score,
            chunk_size=30,  # Process 30 quantiles per chunk
            num_workers=6   # Use 6 parallel workers
        )
    except Exception:
        return {'F1_PA': 0.0, 'P_PA': 0.0, 'R_PA': 0.0}

def _compute_affiliation(labels, score):
    grader = basic_metricor()
    try:
        return grader.metric_Affiliation(labels, score)
    except Exception:
        return 0.0, 0.0, 0.0

def _compute_t_score(labels, score):
    grader = basic_metricor()
    try:
        return grader.metric_F1_T(labels, score)
    except Exception:
        return {'F1_T': 0.0, 'P_T': 0.0, 'R_T': 0.0}

def _compute_f1_t(labels, score):
    grader = basic_metricor()
    try:
        # Use non-parallel path here to avoid pickling issues inside thread workers
        # metric_F1_T(use_parallel=False) runs in-process and returns a dict
        return grader.metric_F1_T(labels, score, use_parallel=True)
    except Exception:
        # Always return a dict to keep downstream code consistent
        return {'F1_T': 0.0, 'P_T': 0.0, 'R_T': 0.0}

def _run_task(func, args):
    return func(*args)


def get_metrics_optimized(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    """
    Fully optimized metrics computation with proper parallelization
    """
    metrics = {}
    start_total = time.time()
    
    # Ensure proper data types to avoid float/integer issues
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    
    # Determine optimal number of workers based on CPU count and workload
    n_cores = multiprocessing.cpu_count()
    
    # For threshold-iterating functions (affiliation and F1_T)
    # Use more workers since they have heavy loops
    heavy_workers = min(n_cores - 2, 8)  # Leave some cores for system
    
    # For simple metrics
    light_workers = min(n_cores // 2, 8)
    
    print(f"Using {heavy_workers} workers for heavy metrics, {light_workers} for light metrics")
    
    # Start the heavy computations first (they take longest)
    print("Starting heavy computations (Affiliation and F1_T)...")
    heavy_start = time.time()
    grader = basic_metricor() 
    with ProcessPoolExecutor(max_workers=2) as main_executor:
        # Launch the two heaviest computations with their own internal parallelization
        affiliation_future = main_executor.submit(
            grader._compute_affiliation_parallel, 
            labels, 
            score, 
            num_workers=heavy_workers
        )
        
        # t_score_future = main_executor.submit(
        #     grader.metric_F1_T_fast,
        #     labels,
        #     score,
        #     num_workers=heavy_workers*2
        # )
        #
        # While heavy computations are running, compute light metrics
        print("Computing light metrics in parallel...")
        light_start = time.time()
        
        with ThreadPoolExecutor(max_workers=light_workers) as light_executor:
            light_futures = {
                'auc_roc': light_executor.submit(_compute_auc_roc, labels, score),
                'auc_pr': light_executor.submit(_compute_auc_pr, labels, score),
                'vus': light_executor.submit(_compute_vus, labels, score, slidingWindow, version),
                'pointf1': light_executor.submit(_compute_pointf1, labels, score),
                'pointf1pa': light_executor.submit(_compute_pointf1pa, labels, score),
                'f1_t': light_executor.submit(_compute_f1_t, labels, score)
            }
            
            # Collect light metric results as they complete
            light_results = {}
            for name, future in light_futures.items():
                try:
                    light_results[name] = future.result()
                    print(f"  ✓ {name} completed")
                except Exception as e:
                    print(f"  ✗ {name} failed: {e}")
                    light_results[name] = None
        
        print(f"Light metrics completed in {time.time() - light_start:.2f}s")
        
        # Wait for heavy computations to complete
        print("Waiting for heavy computations...")
        
        try:
            Affiliation_F, Affiliation_P, Affiliation_R = affiliation_future.result()
            print(f"  ✓ Affiliation completed")
        except Exception as e:
            print(f"  ✗ Affiliation failed: {e}")
            Affiliation_F, Affiliation_P, Affiliation_R = 0.0, 0.0, 0.0
        
        # try:
        #     T_score = t_score_future.result()
        #     print(f"  ✓ F1_T completed")
        # except Exception as e:
        #     print(f"  ✗ F1_T failed: {e}")
        #     T_score = {'F1_T': 0.0, 'P_T': 0.0, 'R_T': 0.0}
    
    print(f"Heavy metrics completed in {time.time() - heavy_start:.2f}s")
    
    # Unpack light results
    AUC_ROC = light_results.get('auc_roc', 0.0)
    AUC_PR = light_results.get('auc_pr', 0.0)
    VUS_result = light_results.get('vus', (0.0, 0.0))
    if isinstance(VUS_result, tuple):
        VUS_ROC, VUS_PR = VUS_result
    else:
        VUS_ROC, VUS_PR = 0.0, 0.0
    # print("HERE IS POINTF1: ")
    # print(light_results.get('pointf1',)) 
    # sys.exit()
    PointF1 = light_results.get('pointf1', {'F1': 0.0, 'Precision': 0.0, 'Recall': 0.0})
    PointF1PA = light_results.get('pointf1pa', {'F1_PA': 0.0, 'P_PA': 0.0, 'R_PA': 0.0})
    T_score = light_results.get('f1_t', {'F1_T': 0.0, 'P_T': 0.0, 'R_T': 0.0})
    # Safeguard: if upstream returned a tuple (e.g., from an older fallback), coerce to dict
    if isinstance(T_score, tuple):
        try:
            T_score = {'F1_T': T_score[0], 'P_T': T_score[1], 'R_T': T_score[2]}
        except Exception:
            T_score = {'F1_T': 0.0, 'P_T': 0.0, 'R_T': 0.0}
    
    # Build final metrics dictionary
    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC
    
    metrics['Standard-F1'] = PointF1.get('F1', 0.0)
    metrics['Standard-Precision'] = PointF1.get('Precision', 0.0)
    metrics['Standard-Recall'] = PointF1.get('Recall', 0.0)
    
    metrics['PA-F1'] = PointF1PA.get('F1_PA', 0.0)
    metrics['PA-Precision'] = PointF1PA.get('P_PA', 0.0)
    metrics['PA-Recall'] = PointF1PA.get('R_PA', 0.0)
    
    metrics['Affiliation-F'] = Affiliation_F
    metrics['Affiliation-P'] = Affiliation_P
    metrics['Affiliation-R'] = Affiliation_R
    
    metrics['F1_T'] = T_score.get('F1_T', 0.0)
    metrics['Precision_T'] = T_score.get('P_T', 0.0)
    metrics['Recall_T'] = T_score.get('R_T', 0.0)
    
    print(f"\nTotal computation time: {time.time() - start_total:.2f}s")
    
    return metrics


def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
    metrics = {}

    # Ensure proper data types to avoid float/integer issues
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)

    '''
    Threshold Independent
    '''
    grader = basic_metricor()
    # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
    try:
        AUC_ROC = grader.metric_ROC(labels, score)
    except Exception:
        AUC_ROC = 0.0
    try:
        AUC_PR = grader.metric_PR(labels, score)
    except Exception:
        AUC_PR = 0.0

    # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
    try:
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, )
    except Exception:
        VUS_ROC, VUS_PR = 0.0, 0.0

    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    PointF1 = grader.metric_standard_F1(labels, score,)
    PointF1PA = grader.metric_PointF1PA(labels, score,)
    # EventF1PA = grader.metric_EventF1PA(labels, score,)
    # RF1 = grader.metric_RF1(labels, score,)
    try:
        Affiliation_F, Affiliation_P, Affiliation_R  = grader.metric_Affiliation(labels, score)
    except Exception:
        Affiliation_F, Affiliation_P, Affiliation_R = 0.0, 0.0, 0.0
    T_score = grader.metric_F1_T(labels, score)

    metrics['AUC-PR'] = AUC_PR
    metrics['AUC-ROC'] = AUC_ROC
    metrics['VUS-PR'] = VUS_PR
    metrics['VUS-ROC'] = VUS_ROC

    metrics['Standard-F1'] = PointF1['F1']
    metrics['Standard-Precision'] = PointF1['Precision']
    metrics['Standard-Recall'] = PointF1['Recall']
    metrics['PA-F1'] = PointF1PA['F1_PA']
    metrics['PA-Precision'] = PointF1PA['P_PA']
    metrics['PA-Recall'] = PointF1PA['R_PA']
    # metrics['Event-based-F1'] = EventF1PA
    # metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    metrics['Affiliation-P'] = Affiliation_P
    metrics['Affiliation-R'] = Affiliation_R

    metrics['F1_T'] = T_score['F1_T']
    metrics['Precision_T'] = T_score['P_T']
    metrics['Recall_T'] = T_score['R_T']

    return metrics


def get_metrics_pred(score, labels, pred, slidingWindow=100):
    metrics = {}

    # Ensure proper data types to avoid float/integer issues
    labels = np.asarray(labels, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)

    grader = basic_metricor()

    PointF1 = grader.standard_F1(labels, score, preds=pred)
    PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
    EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
    RF1 = grader.metric_RF1(labels, score, preds=pred)
    Affiliation_F, Affiliation_P, Affiliation_R = grader.metric_Affiliation(labels, score, preds=pred)
    VUS_R, VUS_P, VUS_F = grader.metric_VUS_pred(labels, preds=pred, windowSize=slidingWindow)

    metrics['Standard-F1'] = PointF1['F1']
    metrics['Standard-Precision'] = PointF1['Precision']
    metrics['Standard-Recall'] = PointF1['Recall']
    metrics['PA-F1'] = PointF1PA
    metrics['Event-based-F1'] = EventF1PA
    metrics['R-based-F1'] = RF1
    metrics['Affiliation-F'] = Affiliation_F
    metrics['Affiliation-P'] = Affiliation_P
    metrics['Affiliation-R'] = Affiliation_R

    metrics['VUS-Recall'] = VUS_R
    metrics['VUS-Precision'] = VUS_P
    metrics['VUS-F'] = VUS_F

    return metrics

def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape) > 1:
        return 0
    if rank == 0:
        return 1
    data = data[: min(20000, len(data))]

    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]

    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]  # Ascending order
        max_local_max = sorted_local_max[0]  # Default
        if rank == 1:
            max_local_max = sorted_local_max[0]
        if rank == 2:
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    max_local_max = i
                    break
        if rank == 3:
            id_tmp = 1
            for i in sorted_local_max[1:]:
                if i > sorted_local_max[0]:
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]:
                    max_local_max = i
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max] < 3 or local_max[max_local_max] > 300:
            return 125
        return local_max[max_local_max] + base
    except Exception:
        return 125