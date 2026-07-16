import os
import numpy as np

def split_into_batches(lst, batch_size, warmup_batch_size=None):
    warmup_batch_size = batch_size if warmup_batch_size is None else warmup_batch_size
    warmup_batch, rest = lst[0:warmup_batch_size], lst[warmup_batch_size:]
    batches = [rest[i:i + batch_size] for i in range(0, len(rest), batch_size)]
    return [warmup_batch] + batches

def split_data_by_arcs(acp_config, arcs):
    arc_keyword = 'arc_name' if 'arc_name' in acp_config.dtype.names else 'arc'
    unique_arcs = list(set(acp_config[arc_keyword].astype('U20'))) or arcs
    unique_arcs = sorted(unique_arcs)

    arc_data = {}
    for arc in unique_arcs:
        arc_data[arc] = acp_config[acp_config[arc_keyword].astype('U20') == arc]
    return arc_data

def check_missing_ground_truth_dose_files(pat_name, acp_config, dtd_load_dir, arcs=None):
    results = {}
    split_arc_data = split_data_by_arcs(acp_config, arcs)

    for arc, arc_acp_config in split_arc_data.items():

        arc_missing_doses = []
        arc_present_doses = []
        arc_cp_indices = list(range(1, len(arc_acp_config)))

        for cp in arc_cp_indices:
            name = f'{pat_name}_arc{arc}_CP{cp}_DTD.npz'
            if not os.path.exists(os.path.join(dtd_load_dir, name)):
                arc_missing_doses.append(name)
            else:
                arc_present_doses.append(name)
        
        arc_results = {
            'num_total_control_points' : len(arc_cp_indices),
            'num_available_ground_truth_doses' : len(arc_present_doses),
            'missing_doses' : arc_missing_doses,
            'present_doses' : arc_present_doses
        }

        results[arc] = arc_results
    
    return results

def generate_full_plan_monte_carlo_doses(pat_name, dtd_load_dir, acp_config, dtd_scale_factor=1.0, scale_by_metersets=True, arcs=None):

    results = {}
    split_arc_data = split_data_by_arcs(acp_config, arcs)

    total_dose = 0.0 * np.load(os.path.join(dtd_load_dir, os.listdir(dtd_load_dir)[0]))['dose']

    for arc, arc_acp_config in split_arc_data.items():
        
        arc_dose = 0.0 * np.load(os.path.join(dtd_load_dir, os.listdir(dtd_load_dir)[0]))['dose']

        for acp_row in arc_acp_config:

            cp_idx = acp_row['cp_idx']
            if str(cp_idx) == '0':
                continue
                
            name = f'{pat_name}_arc{arc}_CP{cp_idx}_DTD.npz'
            dtd_path = os.path.join(dtd_load_dir, name)

            if os.path.exists(dtd_path):
                dtd = np.load(dtd_path)['dose'] 
                
                if np.isnan(dtd).any():
                    print(f'{name} has NaNs!')
                    continue

                scale_factor = dtd_scale_factor * acp_row['CMU'] * 0.01 if scale_by_metersets else 1.0

                dtd = dtd * scale_factor
                arc_dose += dtd
                total_dose += dtd
            else:
                print(f'{name} not found')
        
        results[f'arc{arc}'] = arc_dose
        print(f'Arc {arc} max dose: {np.max(arc_dose)}')
    
    results['full_plan_dose'] = total_dose
    print(f'Full plan max dose: {np.max(total_dose)}')

    return results
