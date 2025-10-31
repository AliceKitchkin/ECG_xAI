import neurokit2 as nk
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import matplotlib.ticker as mticker


class SingleBeatsDetector:
    def __init__(self, sampling_rate=100, method='neurokit', lead_for_rpeak_detection=1):
        """
        Initialize SingleBeatsDetector with default parameters.
        
        Args:
            sampling_rate: Sampling rate of ECG signals
            method: Method for R-peak detection and cleaning
            lead_for_rpeak_detection: Lead index for R-peak detection (usually Lead II = 1)
            n_workers: Number of workers for parallel processing
        """
        self.sampling_rate = sampling_rate
        self.method = method
        self.lead_for_rpeak_detection = lead_for_rpeak_detection
        
        # Default lead names
        self.lead_names = ['I', 'II', 'III', 'AVL', 'AVR', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # ---------------- CLEANING & R-PEAK DETECTION ----------------
    @staticmethod
    def process_single_signal(args, y_relabel, CLASS_NAMES):
        """
        Einzelnes Signal verarbeiten
        """
        signal_idx, signal, sampling_rate, method, lead_for_rpeak_detection = args
        try:
            np.random.seed(42 + signal_idx)
            info = {}
            info["signal_idx"] = signal_idx
            info["label_idx"] = int(y_relabel[signal_idx])
            info["label_name"] = CLASS_NAMES[info["label_idx"]]

            lead_from_signal = signal[:, lead_for_rpeak_detection]
            signals_processed, rpeak_info = nk.ecg_process(
                lead_from_signal, sampling_rate=sampling_rate, method=method
            )
            rpeaks = rpeak_info["ECG_R_Peaks"]
            for k, v in rpeak_info.items():
                info[k] = v
            n_leads = signal.shape[1]
            cleaned_leads = np.zeros_like(signal)
            for lead_idx in range(n_leads):
                lead_data = signal[:, lead_idx]
                cleaned_signal = nk.ecg_clean(lead_data, sampling_rate=sampling_rate, method=method)
                cleaned_leads[:, lead_idx] = cleaned_signal
            return signal_idx, cleaned_leads, rpeaks, info
        except Exception as e:
            return signal_idx, None, np.array([]), None
        
    @staticmethod
    def clean_and_extract_rpeaks_sequential(
        ecg_signal, y_relabel, class_names, sampling_rate, method,
        leads_to_try_for_rpeaks=None, ecg_ids=None, handling=False):
        """
        Cleaning and R-peak detection for ECG signals.

        Args:
            ecg_signal: List of ECG signals (NumPy Arrays)
            y_relabel: One-hot encoded labels
            class_names: List of class names
            sampling_rate: Sampling rate of ECG signals
            method: Method for R-peak detection and cleaning
            leads_to_try_for_rpeaks: List of lead indices to try for R-peak detection
        Returns:
            all_cleaned_signals: List of cleaned ECG signals (NumPy Arrays)
            all_rpeaks: List of R-peak arrays  
            detection_info: List of detection info dictionaries
        """
        n_signals = len(ecg_signal)
        all_cleaned_signals = []
        all_rpeaks = []
        detection_info = []

        for signal_idx in tqdm(range(n_signals), desc="Processing ECG signals"):
            np.random.seed(42 + signal_idx)
            info = {}
            ecg_id = ecg_ids[signal_idx] if (ecg_ids is not None and signal_idx < len(ecg_ids)) else signal_idx
            info['ecg_id'] = ecg_id
            info["signal_idx"] = signal_idx
            info["label_idx"] = int(y_relabel[signal_idx])
            info["label_name"] = class_names[info["label_idx"]]
            signal = ecg_signal[signal_idx]

            # ----- R-Peak Detection with one signal lead -----
            rpeak_success = False
            for lead_try in leads_to_try_for_rpeaks:
                try:    
                    lead_from_signal = signal[:, lead_try]
                    one_signal_processed, rpeak_info = nk.ecg_process(
                        lead_from_signal, sampling_rate=sampling_rate, method=method
                    )
                    rpeaks = rpeak_info["ECG_R_Peaks"]
                    for k, v in rpeak_info.items():
                        info[k] = v
                    info['leads_for_rpeak_detection'] = lead_try
                    rpeak_success = True
                    break  # Erfolg, Schleife verlassen
                except Exception as e:
                    print(f"Error in R-peak detection for signal {signal_idx} with lead {lead_try}: {e}")
                    continue

            if not rpeak_success:
                all_cleaned_signals.append(None)
                all_rpeaks.append(np.array([]))
                detection_info.append(None)
                continue # signal will be skipped
                
            # ----- Clean all 12 leads -----
            try:
                cleaned_leads = np.zeros_like(signal) # for initialization
                for lead_idx in range(12): # 12 leads
                    lead_data = signal[:, lead_idx]
                    cleaned_signal = nk.ecg_clean(lead_data, sampling_rate=sampling_rate, method=method)
                    cleaned_leads[:, lead_idx] = cleaned_signal
                # Store results
                all_cleaned_signals.append(cleaned_leads)
                all_rpeaks.append(rpeaks)
                detection_info.append(info)
            except Exception as e:
                print(f"Error in cleaning for signal {signal_idx}: {e}")
                all_cleaned_signals.append(None)
                all_rpeaks.append(np.array([]))
                detection_info.append(None)



        return all_cleaned_signals, all_rpeaks, detection_info

    @staticmethod
    def save_clean_signals_and_rpeaks(all_cleaned_signals, all_rpeaks, detection_info,
                                      output_dir, sampling_rate=100, method='neurokit'):
        """
        Speichert R-Peak Detection Ergebnisse als Dateien.
        
        Args:
            all_cleaned_signals: List of cleaned ECG signals (NumPy Arrays)
            all_rpeaks: List of R-peak arrays  
            detection_info: List of detection info dictionaries
            output_dir: Ausgabeordner
            sampling_rate: Sampling rate
            method: R-peak detection method
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. build list of valid signals and their indices
        valid_signal_indices = []
        signals_list = []
        for i, signal in enumerate(all_cleaned_signals):
            if signal is not None:
                # Stelle sicher, dass es ein NumPy Array ist
                if isinstance(signal, list):
                    signal = np.array(signal)
                signals_list.append(signal)
                valid_signal_indices.append(i)
        
        # 2. Save all_cleaned_signals as single NumPy array (only valid signals)
        if signals_list:
            with tqdm(total=1, desc="Saving all_cleaned_signals.npy") as pbar:
                all_signals_array = np.stack(signals_list, axis=0)
                np.save(os.path.join(output_dir, 'all_cleaned_signals.npy'), all_signals_array)
                pbar.update(1)
        
        # 3. Save all_rpeaks as Pickle
        with tqdm(total=1, desc="Saving rpeaks.pkl") as pbar:
            with open(os.path.join(output_dir, 'rpeaks.pkl'), 'wb') as f:
                pickle.dump(all_rpeaks, f)
            pbar.update(1)
        
        # 4. Save detection_info as Pickle
        with tqdm(total=1, desc="Saving detection_info.pkl") as pbar:
            with open(os.path.join(output_dir, 'detection_info.pkl'), 'wb') as f:
                pickle.dump(detection_info, f)
            pbar.update(1)

        # 5. Extract ecg_ids from detection_info (preserve alignment with original list indices)
        ecg_ids = []
        for info in detection_info:
            if isinstance(info, dict):
                ecg_ids.append(info.get('ecg_id'))
            else:
                ecg_ids.append(None)

        # 6. build metadata including ecg_ids for robust mapping on load
        n_signals = len(all_cleaned_signals)
        n_signals_with_rpeaks = sum(
            (signal is not None) and (rpeaks is not None) and hasattr(rpeaks, '__len__') and (len(rpeaks) > 0)
            for signal, rpeaks in zip(all_cleaned_signals, all_rpeaks)
        )

        metadata = {
            'sampling_rate': sampling_rate,
            'method': method,
            'n_signals': n_signals,
            'valid_signal_indices': valid_signal_indices,
            'n_valid_signals': len(valid_signal_indices),
            'n_signals_with_rpeaks': n_signals_with_rpeaks,
            'valid_rpeaks': sum(len(rpeaks) for rpeaks in all_rpeaks if rpeaks is not None),
            'signal_shape': all_signals_array.shape if signals_list else None,
            'storage_method': 'single_file',
            'ecg_ids': ecg_ids,
            'valid_signal_ecg_ids': [ecg_ids[i] for i in valid_signal_indices]
        }
        
        with tqdm(total=1, desc="Saving metadata.pkl") as pbar:
            with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            pbar.update(1)
        
        print(f"R-Peak results saved to: {output_dir}")
        print(f"   - {metadata['n_signals']} total signals")
        print(f"   - {metadata['n_valid_signals']} valid signals (not None)")
        print(f"   - {metadata['valid_rpeaks']} valid R-peaks (not None)")
        print(f"   - Signal array shape: {metadata['signal_shape']}")

    @staticmethod
    def load_clean_signals_and_rpeaks(input_dir):
        """
        Lädt R-Peak Detection Ergebnisse aus Dateien.
        
        Returns:
            tuple: (all_cleaned_signals, all_rpeaks, detection_info, metadata)
        """
        # 1. Load Metadata
        with tqdm(total=1, desc="Loading metadata.pkl") as pbar:
            with open(os.path.join(input_dir, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            pbar.update(1)
        
        # 2. Checke Storage Method
        storage_method = metadata.get('storage_method', 'individual_files')
        
        if storage_method == 'single_file' or os.path.exists(os.path.join(input_dir, 'all_cleaned_signals.npy')):
            all_signals_array = np.load(os.path.join(input_dir, 'all_cleaned_signals.npy'))
            
            # reconstruct original list shape
            all_cleaned_signals = [None] * metadata['n_signals']
            with tqdm(total=len(metadata['valid_signal_indices']), desc="Reconstructing signal list") as pbar:
                for idx, signal_idx in enumerate(metadata['valid_signal_indices']):
                    all_cleaned_signals[signal_idx] = all_signals_array[idx]
                    pbar.update(1)
        else:
            print(f"Unsupported storage method")
            return None, None, None, None
        
        # 3. Load R-Peaks
        with tqdm(total=1, desc="Loading rpeaks.pkl") as pbar:
            with open(os.path.join(input_dir, 'rpeaks.pkl'), 'rb') as f:
                all_rpeaks = pickle.load(f)
            pbar.update(1)
        
        # 4. Load Detection Info
        with tqdm(total=1, desc="Loading detection_info.pkl") as pbar:
            with open(os.path.join(input_dir, 'detection_info.pkl'), 'rb') as f:
                detection_info = pickle.load(f)
            pbar.update(1)
        
        print(f"R-Peak results loaded from: {input_dir}")
        print(f"   - {metadata['n_signals']} total signals")
        print(f"   - {metadata['n_valid_signals']} valid signals (not None)")
        print(f"   - {metadata['valid_rpeaks']} valid R-peaks (not None)")
        print(f"   - Signal array shape: {metadata['signal_shape']}")
        
        return all_cleaned_signals, all_rpeaks, detection_info, metadata


    # ---------------- EDGE CASE HANDLING ----------------
    @staticmethod
    def print_failed_rpeaks_overview(signals, rpeaks, detection_info, mad_thresh=4.0):
        """
        Gibt eine Übersicht über fehlerhafte R-Peaks (z.B. Amplitude < threshold) aus.
        """
        signalids_none = []
        rpeaks_none = []
        failed_rpeaks = []
        failed_norm = []
        failed_mi = []
        n_failed_total = 0
        n_failed_norm_total = 0
        n_failed_mi_total = 0

        def _orig_id(info, idx):
            if info is None:
                return idx
            try:
                return info.get("signal_idx", idx)
            except Exception:
                return idx
        
        # Loop through signals and rpeaks
        for idx, (signal, rpeaks_arr, info) in enumerate(zip(signals, rpeaks, detection_info)):
            orig_id = _orig_id(info, idx)
            if signal is None:
                signalids_none.append(orig_id)
                continue
            if rpeaks_arr is None or not hasattr(rpeaks_arr, '__len__') or len(rpeaks_arr) == 0:
                rpeaks_none.append(orig_id)
                continue

            # determine which lead was used for R-Peak Detection
            leads_used = info.get('leads_for_rpeak_detection') or info.get('used_leads')
            if leads_used is None:
                leads_used = [1]  # fallback to Lead II
            elif isinstance(leads_used, int):
                leads_used = [leads_used]
        
            # Check if at least one rpeak-amplitude is below threshold
            failed_mask = SingleBeatsDetector.identify_failed_rpeaks(signal, rpeaks_arr, leads_used, mad_thresh)

            # Count failed rpeaks
            n_failed = np.sum(failed_mask)
            n_failed_total += n_failed
            
            if n_failed > 0:
                failed_rpeaks.append(orig_id)
                if info is not None:
                    label = info.get('label_name', 'UNKOWN')
                    if label == 'NORM':
                        failed_norm.append(orig_id)
                        n_failed_norm_total += n_failed
                    elif label == 'MI':
                        failed_mi.append(orig_id)
                        n_failed_mi_total += n_failed

        # Calculate statistics
        nr_signals = len(signals)
        nr_rpeaks = sum(len(rp) for rp, sig in zip(rpeaks, signals) if sig is not None and hasattr(rp, '__len__'))
        nr_failed_rpeaks = len(failed_rpeaks)
        nr_valid_rpeaks = nr_rpeaks - n_failed_total

        samples = 20

        print(f"> Signals None: {len(signalids_none)}")
        print(f"> ID Examples: {signalids_none[:samples]}\n")

        print(f"> Signals with R-Peaks None: {len(rpeaks_none)}")
        print(f"> ID Examples: {rpeaks_none[:samples]}\n")

        print(f"Found R-Peaks: {nr_rpeaks} (100%)")
        avg_rpeaks = nr_rpeaks / nr_signals if nr_signals > 0 else 0
        print(f"Avg R-Peaks per Signal: {avg_rpeaks:.2f}\n")

        print(f"> Signals with at least one failed R-Peaks: {nr_failed_rpeaks}")
        print(f"> ID Examples: {failed_rpeaks[:samples]}\n")

        valid_rpeaks_perc = nr_valid_rpeaks / nr_rpeaks * 100 if nr_rpeaks > 0 else 0
        print(f"> Valid R-Peaks: {nr_valid_rpeaks} ({valid_rpeaks_perc:.2f}%)")

        failed_rpeak_perc = n_failed_total / nr_rpeaks * 100 if nr_rpeaks > 0 else 0
        print(f"> Failed R-Peaks: {n_failed_total} ({failed_rpeak_perc:.2f}%)")

        norm_failed_perc = n_failed_norm_total / n_failed_total * 100 if n_failed_total > 0 else 0
        print(f"\t> In NORM:: {n_failed_norm_total} ({norm_failed_perc:.2f}%)")
        print(f"\t> ID Examples: {failed_norm[:samples]}")

        mi_failed_perc = n_failed_mi_total / n_failed_total * 100 if n_failed_total > 0 else 0
        print(f"\t> In MI:: {n_failed_mi_total} ({mi_failed_perc:.2f}%)")
        print(f"\t> ID Examples: {failed_mi[:samples]}")


    @staticmethod
    def detect_rpeaks(signal, lead_idx, sampling_rate=100, method="neurokit"):
        """Detect R-peaks on a specific lead."""
        lead = signal[:, lead_idx]
        _, detection_info = nk.ecg_process(lead, sampling_rate=sampling_rate, method=method)
        return detection_info
    
    @staticmethod
    def identify_failed_rpeaks(signals, rpeaks_list, leads_used_list, mad_thresh=4.0):
        """
        Identifies failed R-peaks for one or multiple signals.

        Args:
            signals: np.ndarray (single signal) or list of signals
            rpeaks_list: array-like (single) or list of arrays
            leads_used_list: int/list (single) or list of leads
            mad_thresh: threshold for outlier detection

        Returns:
            failed_masks: boolean array (single) or list of boolean arrays
        """
        # Handle single signal case
        if not isinstance(signals, list):
            signals = [signals]
            rpeaks_list = [rpeaks_list]
            leads_used_list = [leads_used_list]

        failed_masks = []
        for signal, rpeaks, leads_used in zip(signals, rpeaks_list, leads_used_list):
            if signal is None or len(rpeaks) == 0 or leads_used is None:
                failed_masks.append(np.array([], dtype=bool))
                continue
            if isinstance(leads_used, int):
                leads_used = [leads_used]
            amplitudes = np.stack([signal[rpeaks, lead] for lead in leads_used], axis=1)
            failed_mask = np.zeros(len(rpeaks), dtype=bool)
            for i in range(amplitudes.shape[1]):
                lead_amps = amplitudes[:, i]
                median = np.median(lead_amps)
                mad = np.median(np.abs(lead_amps - median))
                if mad == 0:
                    mad = 1e-8
                outlier_mask = np.abs(lead_amps - median) > mad_thresh * mad
                failed_mask = failed_mask | outlier_mask
            failed_masks.append(failed_mask)
        return failed_masks if len(failed_masks) > 1 else failed_masks[0]
    
    @staticmethod
    def choose_best_rpeaks(signal, rpeaks_candidates, lead_indices, mad_thresh=4.0):
        best_idx = None
        best_score = None
        for i, (rpeaks, lead_idx) in enumerate(zip(rpeaks_candidates, lead_indices)):
            if len(rpeaks) == 0:
                continue
            failed_mask = SingleBeatsDetector.identify_failed_rpeaks(signal, rpeaks, lead_idx, mad_thresh)
            n_bad = np.sum(failed_mask)
            amps = signal[rpeaks, lead_idx]
            mean_amp = np.mean(amps)
            score = (n_bad, -mean_amp)
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
        if best_idx is not None:
            return rpeaks_candidates[best_idx], lead_indices[best_idx]
        else:
            return np.array([]), None
    
    @staticmethod
    def handle_failed_rpeaks(
        signals, rpeaks_list, detection_info, new_leads_to_try=[10, 11, 1],
        mad_thresh=11, sampling_rate=100, ecg_ids=None):
        """
        Fixes failed R-peaks by trying alternative leads and replacing only failed peaks or the whole array.
        Args:
            signals: List of ECG signals (NumPy Arrays)
            rpeaks_list: List of R-peak arrays  
            detection_info: List of detection info dictionaries
            new_leads_to_try: List of lead indices to try for R-peak detection
            mad_thresh: Threshold for outlier detection
            sampling_rate: Sampling rate of ECG signals
            ecg_ids: Optional list of ECG IDs for robust mapping
        Returns:
            Updated rpeaks_list and detection_info with fixed R-peaks
        """
        n_single_replaced = 0
        n_array_replaced = 0
        single_replaced_signals = []
        array_replaced_signals = []
        fields_to_update = ["ECG_Raw", "ECG_Clean", "ECG_R_Peaks", "ECG_Rate", "ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks",
                            "ECG_T_Peaks", "ECG_P_Onsets", "ECG_P_Offsets", "ECG_T_Onsets", "ECG_T_Offsets",
                            "ECG_R_Onsets", "ECG_R_Offsets", "ECG_Phase_Atrial", "ECG_Phase_Ventricular",
                            "ECG_Atrial_PhaseCompletion", "ECG_Ventricular_PhaseCompletion", "ECG_Peaks"]
        
        # 0. Ensure each detection_info entry contains persistent original ecg_id
        if ecg_ids is not None:
            for i in range(len(detection_info)):
                info = detection_info[i]
                if info is None:
                    # create minimal dict to hold ecg_id (keeps shape consistent)
                    detection_info[i] = {"ecg_id": ecg_ids[i] if i < len(ecg_ids) else None}
                else:
                    # only set if missing
                    if "ecg_id" not in info or info.get("ecg_id") is None:
                        info["ecg_id"] = ecg_ids[i] if i < len(ecg_ids) else None
        else:
            # ensure key exists (may remain None) for downstream code that expects a dict
            for i in range(len(detection_info)):
                info = detection_info[i]
                if info is None:
                    detection_info[i] = {"ecg_id": None}
                else:
                    info.setdefault("ecg_id", None)

        # 1. Identify Signals with failed R-peaks
        leads_used_list = []
        for info in detection_info:
            if info is None:
                leads_used_list.append(None)
            else:
                leads_used = info.get('leads_for_rpeak_detection') or info.get('used_leads')
                leads_used_list.append(leads_used)
        failed_masks = SingleBeatsDetector.identify_failed_rpeaks(signals, rpeaks_list, leads_used_list, mad_thresh)
        failed_indices = [idx for idx, mask in enumerate(failed_masks) if mask is not None and np.any(mask)]

        # 2. Loop through signals with failed R-peaks
        for idx in tqdm(failed_indices, desc="Handling Signals with failed R-peaks", total=len(failed_indices)):
            signal = signals[idx]
            rpeaks = rpeaks_list[idx]
            
            # Collect R-Peak candidates from other leads
            rpeaks_candidates = []
            lead_indices = []
            for lead in new_leads_to_try:
                try:
                    new_detection_info = SingleBeatsDetector.detect_rpeaks(signal, lead, sampling_rate)
                    new_rpeaks = new_detection_info["ECG_R_Peaks"]
                    rpeaks_candidates.append(new_rpeaks)
                    lead_indices.append(lead)
                except Exception:
                    continue
            
            # Choose the best lead based on criteria
            best_rpeaks, best_lead = SingleBeatsDetector.choose_best_rpeaks(signal, rpeaks_candidates, lead_indices, mad_thresh)
            if len(best_rpeaks) == 0:
                continue

            leads_used = detection_info[idx].get('leads_for_rpeak_detection') or detection_info[idx].get('used_leads')
            if leads_used is None:
                leads_used = [1]
            elif isinstance(leads_used, int):
                leads_used = [leads_used]
            
            if len(best_rpeaks) == len(rpeaks):
                # Replace only failed R-peaks
                rpeaks_fixed = np.array(rpeaks, copy=True)
                failed_mask = SingleBeatsDetector.identify_failed_rpeaks(signal, rpeaks, leads_used, mad_thresh)
                n_replaced = np.sum(failed_mask) # count replaced rpeaks

                if n_replaced > 0:
                    rpeaks_fixed[failed_mask] = best_rpeaks[failed_mask] # Replace only failed
                    rpeaks_list[idx] = rpeaks_fixed # Update in main list
                    n_single_replaced += n_replaced # Update counters and logs
                    single_replaced_signals.append((idx, n_replaced)) # Log the replacement

                    # --- Update detection_info fields for replaced indices ---
                    try:
                        new_detection_info = SingleBeatsDetector.detect_rpeaks(signal, best_lead, sampling_rate)
                        for key in fields_to_update:
                            if key in new_detection_info:
                                detection_info[idx][key] = new_detection_info[key]
                    except Exception as e:
                        print(f"Warning: Could not update detection_info for signal {idx}: {e}")
            else:
                # Replace the whole array
                rpeaks_list[idx] = best_rpeaks
                try:
                    new_detection_info = SingleBeatsDetector.detect_rpeaks(signal, best_lead, sampling_rate)
                    for key in fields_to_update:
                        if key in new_detection_info:
                            detection_info[idx][key] = new_detection_info[key]
                except Exception as e:
                    print(f"Warning: Could not update detection_info for signal {idx}: {e}")
                detection_info[idx]["ECG_R_Peaks"] = best_rpeaks
                n_array_replaced += 1
                array_replaced_signals.append(idx)
            
            # update detection_info
            if best_lead is not None:
                if "used_leads" in detection_info[idx]:
                    if best_lead not in detection_info[idx]["used_leads"]:
                        detection_info[idx]["used_leads"].append(best_lead)
                else:
                    detection_info[idx]["used_leads"] = [best_lead]
        
        print(20*'-')
        print(f"Number of single replaced R-Peaks: {n_single_replaced}")
        if single_replaced_signals:
            print("Single replaced R-Peaks:")
            # Gruppiere nach Anzahl der Ersetzungen
            grouped = defaultdict(list)
            for idx, n in single_replaced_signals:
                grouped[n].append(idx)
            for n in sorted(grouped):
                print(f"({n}): ID examples: {grouped[n][:10]}")

        print(20*'-')
        print(f"Number of completely replaced R-Peak arrays: {n_array_replaced}")
        if array_replaced_signals:
            print(f"ID examples: {array_replaced_signals[:10]}")

        return signals, rpeaks_list, detection_info
    
 
    # ------------------------ SEGMENTATION ------------------------
    @staticmethod
    def segment_single_signal(signal_idx, cleaned_signal, rpeaks, sampling_rate, epochs_start, epochs_end):
        """
        Einzelnes Signal segmentieren - Standalone Version ohne Threading.
        """
        try:
            if cleaned_signal is None or len(rpeaks) == 0:
                return signal_idx, None

            signal_beats = {}
            for lead_idx in range(cleaned_signal.shape[1]):
                lead_data = cleaned_signal[:, lead_idx]
                try:
                    epochs = nk.epochs_create(
                        lead_data,
                        events=rpeaks,
                        sampling_rate=sampling_rate,
                        epochs_start=epochs_start,
                        epochs_end=epochs_end
                    )
                    signal_beats[lead_idx] = epochs
                except Exception as e:
                    signal_beats[lead_idx] = {}
            return signal_idx, signal_beats
        except Exception as e:
            return signal_idx, None

    @staticmethod
    def segment_ecg_beats_sequential(cleaned_signals, rpeaks_list, sampling_rate=100, epochs_start=-0.2, epochs_end=0.5):
        """
        Segment ECG signals into individual beats - Sequential Version (ohne Threading).
        
        Args:
            cleaned_signals: List of cleaned ECG signals 
            rpeaks_list: List of R-peak arrays
            sampling_rate: Sampling rate (default: 100)
            epochs_start: Start of epoch relative to R-peak (seconds)
            epochs_end: End of epoch relative to R-peak (seconds)
        
        Returns:
            List of segmented beats for each signal and each lead
        """
        n_signals = len(cleaned_signals)
        all_segmented_beats = [None] * n_signals

        print(f"Segmenting {n_signals} signals sequentially...")
        print(f"Window: {epochs_start}s to {epochs_end}s")

        for signal_idx in tqdm(range(n_signals), desc="Segmenting beats"):
            try:
                signal_idx, signal_beats = SingleBeatsDetector.segment_single_signal(
                    signal_idx,
                    cleaned_signals[signal_idx],
                    rpeaks_list[signal_idx],
                    sampling_rate,
                    epochs_start,
                    epochs_end
                )
                all_segmented_beats[signal_idx] = signal_beats
            except Exception as e:
                print(f"Error segmenting signal {signal_idx}: {e}")
                all_segmented_beats[signal_idx] = None

        valid_signals = sum(1 for beats in all_segmented_beats if beats is not None)
        valid_beats = 0
        n_beats = 0
        for signal_beats in all_segmented_beats:
            n_beats += len(signal_beats[0])
            if signal_beats is not None and 0 in signal_beats:
                valid_beats += len(signal_beats[0])

        print(f"Segmentierung abgeschlossen!")
        print(f"   - {n_signals} total signals")
        print(f"   - {valid_signals} valid signals (not None)")

        print(f"   - {n_beats} total beats")
        print(f"   - {valid_beats} valid beats (not None)")

        return all_segmented_beats

    # SAVING & LOADING
    @staticmethod
    def save_segmented_beats(all_segmented_beats, output_dir, epochs_start=-0.2,
                         epochs_end=0.5, sampling_rate=100, method='pantompkins1985'):
        """
        Speichert segmentierte Beats als Dateien.
        
        Args:
            all_segmented_beats: List of segmented beats for each signal
            output_dir: Ausgabeordner
            epochs_start: Start des Fensters (seconds)
            epochs_end: Ende des Fensters (seconds) 
            sampling_rate: Sampling rate
            method: Segmentierungsmethode
        """
        # Erstelle Ausgabeordner
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Segmented Beats als NumPy Array speichern
        with tqdm(total=1, desc="Saving segmented_beats.npy") as pbar:
            segmented_array = np.array(all_segmented_beats, dtype=object)
            np.save(os.path.join(output_dir, 'segmented_beats.npy'), segmented_array)
            pbar.update(1)
        
        # 2. Metadata speichern
        total_beats = 0
        valid_signals = 0
        for signal_beats in all_segmented_beats:
            if signal_beats is not None:
                valid_signals += 1
                if 0 in signal_beats:
                    total_beats += len(signal_beats[0])
            
        metadata = {
            'epochs_start': epochs_start,
            'epochs_end': epochs_end,
            'sampling_rate': sampling_rate,
            'method': method,
            'n_signals': len(all_segmented_beats),
            'valid_signals': valid_signals,
            'total_beats': total_beats,
            'window_length_samples': int((epochs_end - epochs_start) * sampling_rate)
        }
        
        with tqdm(total=1, desc="Saving segmentation_metadata.pkl") as pbar:
            with open(os.path.join(output_dir, 'segmentation_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
            pbar.update(1)

        print(f"Segmented beats saved to: {output_dir}")
        print(f"   - {metadata['n_signals']} total signals")
        print(f"   - {metadata['valid_signals']} valid signals") 
        print(f"   - {metadata['total_beats']} total beats")
        print(f"   - Window: {epochs_start}s to {epochs_end}s")

    @staticmethod
    def load_segmented_beats(input_dir):
        """
        Lädt segmentierte Beats aus Dateien.
        """
        # Lade segmentierte Beats als NumPy Array mit tqdm
        with tqdm(total=1, desc="Loading segmented_beats.npy") as pbar:
            all_segmented_beats = np.load(os.path.join(input_dir, 'segmented_beats.npy'), allow_pickle=True)
            pbar.update(1)

        # Lade Metadata mit tqdm
        with tqdm(total=1, desc="Loading segmentation_metadata.pkl") as pbar:
            with open(os.path.join(input_dir, 'segmentation_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            pbar.update(1)

        print(f"Segmented beats loaded from: {input_dir}")
        print(f"   - {metadata['n_signals']} total signals")
        print(f"   - {metadata['valid_signals']} valid signals")
        print(f"   - {metadata['total_beats']} total beats") 
        print(f"   - Window: {metadata['epochs_start']}s to {metadata['epochs_end']}s")
        print(f"   - Method: {metadata['method']}")
        print(f"   - Data type: {type(all_segmented_beats)}")  # Debug-Info

        return all_segmented_beats, metadata

    # PLOTTING
    @staticmethod
    def plot_beat_segments(all_segmented_beats, signal_idx=0,
                        detection_info=None, epochs_start=-0.2, epochs_end=0.5,
                        rpeak_marker=0, max_beats_per_lead=15, lead_names=None,
                        sampling_frequency=100, fig_size=(12, 16)):
        """
        Visualisiert die segmentierten Beats für alle 12 Ableitungen eines EKG-Signals.

        Args:
            all_segmented_beats: Liste aller segmentierten Beats
            signal_idx: Index des EKG-Signals aus all_segmented_beats
            max_beats_per_lead: Maximale Anzahl Beats pro Ableitung zur besseren Übersichtlichkeit
            lead_names: Liste der Lead-Namen (optional)
            sampling_frequency: Sampling Rate (Hz)
        """
        if signal_idx >= len(all_segmented_beats) or all_segmented_beats[signal_idx] is None:
            print(f"Signal {signal_idx} ist nicht verfügbar oder fehlerhaft.")
            return
        
        signal_beats = all_segmented_beats[signal_idx]

        # use col "signal_id" in detection info, not index in list
        signal_id = detection_info[signal_idx].get("signal_id", signal_idx)
        label = detection_info[signal_idx]["label_name"]
        
        # could be more then one lead
        # lead names comma separated string
        lead_used_for_rpeaks = detection_info[signal_idx].get("leads_for_rpeak_detection", "Unknown")
        lead_used_for_rpeaks_str = ", ".join([lead_names[i] for i in lead_used_for_rpeaks]) if isinstance(lead_used_for_rpeaks, list) else str(lead_used_for_rpeaks)
        
        # Erstelle Figure mit 12 Subplots
        fig, axes = plt.subplots(12, 1, figsize=fig_size)
        title = (
            f'R-Peaks applied to all 12 ECG Leads\n'
            f'Leads used for R-Peak Detection: {lead_used_for_rpeaks_str}\n'
            f'Signal ID: {signal_idx}, Label: {label}')
        fig.suptitle(title, fontsize=14)

        # Plotte für jede Ableitung (Lead)
        for lead_idx in range(12):
            if lead_idx not in signal_beats:
                axes[lead_idx].text(0.5, 0.5, f'Lead {lead_names[lead_idx]} - No data',
                                    ha='center', va='center', transform=axes[lead_idx].transAxes)
                continue
                
            epochs = signal_beats[lead_idx]

            if not epochs:
                print(f"No epochs for lead {lead_idx} in signal {signal_idx}")
                continue # Wenn keine Beats für diese Ableitung
            
            beat_count = 0
            # Plotte alle Beats dieser Ableitung übereinander
            for beat_key in list(epochs.keys())[:max_beats_per_lead]:
                beat_data = epochs[beat_key].iloc[:, 0]  # Erste Spalte der Beat-Daten
                time_axis = np.arange(len(beat_data)) / sampling_frequency - abs(epochs_start)  # Zeit relativ zu R-Peak

                axes[lead_idx].plot(time_axis, beat_data, alpha=0.6, linewidth=1, color='blue')
                beat_count += 1
            
            # mark rpeak
            axes[lead_idx].axvline(x=rpeak_marker, color='red', linestyle='--', alpha=0.8, linewidth=2, label='R-peak')

            # Achsenbeschriftung und Titel
            #axes[lead_idx].set_title(f'Lead {lead_names[lead_idx]} - {beat_count} Segmented Beats')
            axes[lead_idx].set_ylabel(f'{lead_names[lead_idx]}')
            axes[lead_idx].grid(True, alpha=0.3)
            axes[lead_idx].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
            #axes[lead_idx].set_xlim(epochs_start, epochs_end)
            
            # Nur bei der letzten Ableitung x-Achsen Label
            if lead_idx == 11:
                axes[lead_idx].set_xlabel('Time (s) relative to R-peak')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()


    # ------------------------ PLOTTING ------------------------
    @staticmethod
    def plot_one_signal_with_rpeaks(signal_id, cleaned_signals, rpeaks, detection_info, lead_names, figsize=(15, 20)):
        """
        Plottet ein einzelnes Signal mit allen 12 Ableitungen und R-Peaks.
        
        Args:
            signal_id: Index des zu plottenden Signals
            cleaned_signals: Liste aller bereinigten Signale
            rpeaks: Liste aller R-Peak Arrays
            lead_names: Namen der 12 Ableitungen
            figsize: Größe der Figur
        """
        # Hole Signal und R-Peaks
        cleaned_signal = cleaned_signals[signal_id]
        rpeaks = rpeaks[signal_id]
        label = detection_info[signal_id]["label_name"]
        
        if cleaned_signal is None:
            print(f"Signal {signal_id} ist None - Skip")
            return
        
        # Konvertiere zu NumPy Array falls es eine Liste ist
        if isinstance(cleaned_signal, list):
            try:
                cleaned_signal = np.array(cleaned_signal)
            except Exception as e:
                print(f"Error converting signal {signal_id} to array for plotting: {e}")
                return
        
        # Prüfe ob Signal die richtige Form hat
        if cleaned_signal.ndim != 2 or cleaned_signal.shape[1] != 12:
            print(f"Signal {signal_id} has wrong shape for plotting: {cleaned_signal.shape}, expected (n_samples, 12)")
            return
    
        # Get leads used for R-Peak Detection
        leads_str = "?"
        if detection_info is not None and detection_info[signal_id] is not None:
            leads = detection_info[signal_id].get("leads_for_rpeak_detection")
            if isinstance(leads, list):
                leads_str = ", ".join([lead_names[l] for l in leads])
            elif isinstance(leads, int):
                leads_str = lead_names[leads]
        
        # Erstelle Figure mit 12 Subplots
        fig, axes = plt.subplots(12, 1, figsize=figsize)
        title = (
            f'R-Peaks applied to all 12 ECG Leads\n'
            f'Leads used for R-Peak Detection: {leads_str}\n'
            f'Signal ID: {signal_id}, Label: {label}')
        fig.suptitle(title, fontsize=14)
        
        # Plotte alle 12 Ableitungen
        for lead_idx in range(12):
            lead_data = cleaned_signal[:, lead_idx]
            
            # Plotte bereinigte Signale
            axes[lead_idx].plot(lead_data, color='blue', alpha=0.7, linewidth=1)
            
            # Markiere R-Peaks
            if len(rpeaks) > 0:
                axes[lead_idx].scatter(rpeaks, lead_data[rpeaks], color='red', s=50, marker='o', zorder=5)
            
            # Titel und Labels
            axes[lead_idx].set_ylabel(lead_names[lead_idx])
            axes[lead_idx].grid(True, alpha=0.3)
                    
            # X-Achse nur beim letzten Plot
            if lead_idx == 11:
                axes[lead_idx].set_xlabel('Samples')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.show()

    @staticmethod
    def visualize_heartbeats_overlay(X_beats, y_beats, max_beats=20, figsize=(15, 10), window_config=[-0.2, 0.5]):
        """
        Overlay multiple heartbeats for all 12 leads (similar to old code style)
        
        Args:
            X_beats: Array of single beats
            y_beats: Labels for each beat
            max_beats: Maximum number of beats to overlay
        """
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Create subplots for all 12 leads
        fig, axes = plt.subplots(12, 1, figsize=figsize)
        fig.suptitle("Individual Heart Beats - All 12 Leads", fontsize=14)
        
        # Filter beats by label
        mi_mask = np.array([np.array_equal(label, [1, 0]) for label in y_beats])
        norm_mask = np.array([np.array_equal(label, [0, 1]) for label in y_beats])

        time_axis = np.linspace(window_config[0], window_config[1], X_beats.shape[2])  # Adjust based on your window

        for lead_idx in range(12):
            ax = axes[lead_idx]
            
            # Plot MI beats
            mi_beats = X_beats[mi_mask][:max_beats//2]
            for i, beat in enumerate(mi_beats):
                color = plt.cm.Reds(0.3 + 0.7 * i / len(mi_beats))
                ax.plot(time_axis, beat[lead_idx, :], color=color, alpha=0.5, 
                    label='MI' if i == 0 else "")
            
            # Plot NORM beats
            norm_beats = X_beats[norm_mask][:max_beats//2]
            for i, beat in enumerate(norm_beats):
                color = plt.cm.Blues(0.3 + 0.7 * i / len(norm_beats))
                ax.plot(time_axis, beat[lead_idx, :], color=color, alpha=0.5,
                    label='NORM' if i == 0 else "")
            
            ax.set_title(f"Lead {lead_names[lead_idx]}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='R-peak' if lead_idx == 0 else "")
            ax.grid(True, alpha=0.3)
            
            # Only show legend for first lead to avoid clutter
            if lead_idx == 0:
                ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

    @staticmethod
    def visualize_average_heartbeats_overlay(X_beats, y_beats, max_beats=50, figsize=(15, 30), window_config=[-0.2, 0.5]):
        """
        Show average heartbeats for all 12 leads with overlay of individual beats
        
        Args:
            X_beats: Array of single beats
            y_beats: Labels for each beat
            max_beats: Maximum number of beats to use for averaging
        """
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # Create subplots for all 12 leads
        fig, axes = plt.subplots(12, 1, figsize=figsize)
        fig.suptitle("Average Heart Beats - All 12 Leads", fontsize=14)
        
        # Filter beats by label
        mi_mask = np.array([np.array_equal(label, [1, 0]) for label in y_beats])
        norm_mask = np.array([np.array_equal(label, [0, 1]) for label in y_beats])
        
        # Get beats for averaging (limit to max_beats)
        mi_beats = X_beats[mi_mask][:max_beats//2]
        norm_beats = X_beats[norm_mask][:max_beats//2]
        
        # Calculate averages
        mi_avg = np.mean(mi_beats, axis=0) if len(mi_beats) > 0 else None
        norm_avg = np.mean(norm_beats, axis=0) if len(norm_beats) > 0 else None

        time_axis = np.linspace(window_config[0], window_config[1], X_beats.shape[2])  # Adjust based on your window

        for lead_idx in range(12):
            ax = axes[lead_idx]
            
            # Plot individual MI beats (light/transparent)
            for i, beat in enumerate(mi_beats):
                color = plt.cm.Reds(0.3 + 0.4 * i / len(mi_beats))
                ax.plot(time_axis, beat[lead_idx, :], color=color, alpha=0.2, linewidth=0.5)
            
            # Plot individual NORM beats (light/transparent)
            for i, beat in enumerate(norm_beats):
                color = plt.cm.Blues(0.3 + 0.4 * i / len(norm_beats))
                ax.plot(time_axis, beat[lead_idx, :], color=color, alpha=0.2, linewidth=0.5)
            
            # Plot average lines (bold)
            if mi_avg is not None:
                ax.plot(time_axis, mi_avg[lead_idx, :], color='#c63b3a', linewidth=2.3, 
                    label=f'MI Average')
            
            if norm_avg is not None:
                ax.plot(time_axis, norm_avg[lead_idx, :], color='#539cca', linewidth=2.3,
                    label=f'NORM Average')
            
            ax.set_title(f"Lead {lead_names[lead_idx]}")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='R-peak' if lead_idx == 0 else "")
            ax.grid(True, alpha=0.3)
            
            # Only show legend for first lead to avoid clutter
            if lead_idx == 0:
                ax.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()
