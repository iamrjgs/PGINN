from copy import deepcopy
from collections import defaultdict
import gc
import json
import logging
import os
from os.path import join
from time import perf_counter
import sys

import numpy as np
import pydicom

import math
from numba import cuda
import cupy as cp
_CUDA_DEVICE_NAME = cuda.get_current_device()

from PGINN.raydose.RaydoseComposer import RaydoseComposer
from PGINN.raydose.dicom_processor import (
    get_dose_array_from_dicom_dose,
    substitute_dose_array_in_dicom_dose
)
from PGINN.nn_models.base import ModelFactory
from PGINN.utils.general_functions import split_into_batches
from PGINN.utils.cuda_array_operations import sum_cuda_arrays

class DoseEngine:
    def __init__(self,
                patients,
                base_patient_data_dirpath,
                beam_data_dirpath,
                model_data_dirpath=None,
                model_architechture_name='iDOTA',
                energy=1,
                batch_size=1,
                convert_GyperMU_to_Gy=True,
                reformat_along_beam=True,
                return_to_original_geometry=True,
                num_raytracing_subarc_points=2,
                dtd_load_dir=None,
                dtd_scale_factor=1.0,
                include_rtplan_dose=True,
                save_dirpath=None,
                model_weights_filepath=None,
                model_hyperparams_filepath=None,
                scale_constants_filepath=None,
                pginn_logging_level=logging.INFO,
                raydose_logging_level=logging.WARNING,
                print_model_summary=False,
                individual_cp_save_dir=None,
                arc=None,
                control_points=None
                ):
        # Set pipeline data and settings
        self.patients = patients if isinstance(patients, (list, tuple)) else [patients]
        self.base_patient_data_dirpath = base_patient_data_dirpath
        self.beam_data_dirpath = beam_data_dirpath
        self.model_data_dirpath = model_data_dirpath
        self.model_architechture_name = model_architechture_name
        self.energy = energy
        self.batch_size = batch_size
        self.convert_GyperMU_to_Gy = convert_GyperMU_to_Gy
        self.reformat_along_beam = reformat_along_beam
        self.return_to_original_geometry = return_to_original_geometry
        self.num_raytracing_subarc_points = num_raytracing_subarc_points
        self.dtd_load_dir = dtd_load_dir
        self.dtd_scale_factor = dtd_scale_factor
        self.include_ground_truth = self.dtd_load_dir is not None
        self.include_rtplan_dose = include_rtplan_dose
        self.save_dirpath = save_dirpath
        self.individual_cp_save_dir = individual_cp_save_dir
        self.arc = arc
        self.control_points = control_points
        self.raydose_logging_level = raydose_logging_level
        os.makedirs(self.individual_cp_save_dir, exist_ok=True) if self.individual_cp_save_dir is not None else None
        
        self.set_logger(pginn_logging_level)
        self.give_warnings()

        # Set to use default model included with package if path if not provided
        if model_data_dirpath is None and model_weights_filepath is None:
            import pkg_resources
            model_architechture_name = 'iDOTA'
            model_data_dirpath = pkg_resources.resource_filename('PGINN', 'nn_models/iDOTA/default_model_directory')
            self.logger.warning('Deep learning model path not provided. Loading default iDOTA model.')

        # Load model object, which loads model weights and implements preprocess, infer, and postprocess functions
        self.model = ModelFactory.create(model_architechture_name,
            model_data_dirpath=model_data_dirpath,
            model_weights_path=model_weights_filepath,
            hyperparams_path=model_hyperparams_filepath,
            scale_path=scale_constants_filepath,
            batch_size=self.batch_size
        )

        # Get additional settings from model object
        self.model_weights_filepath = self.model.model_weights_filepath
        self.scale_constants_filepath = self.model.scale_constants_filepath
        self.model_hyperparams_filepath = self.model.model_hyperparams_filepath
        self.scale = self.model.scale
        self.hyperparams = self.model.hyperparams
        self.matrix_size = self.model.input_matrix_size
        self.matrix_size_str = "x".join(map(str, self.matrix_size))
        self.channels = self.model.input_channels
        self.input_batch_shape = self.model.input_batch_shape
        self.batch_cuda_buffer = self.model.batch_cuda_buffer
        self.average_raydoses = self.channels == 2

        self.logger.info(f'PGINN initialized.')
        self.logger.info(f'Using {self.matrix_size_str} model at {self.model_weights_filepath}.')
        self.model.print_summary() if print_model_summary else None

    def initialize_for_patient(self, patient):
        # Initialize results data containers
        self.arc_doses_dict = {}
        self.dicom_objects_dict = {}
        self.arc_runtimes_dict = {}
        self.operation_times = defaultdict(list)

        # Load ray-tracing pipeline
        self.raytracer = RaydoseComposer(
            patient=patient,
            base_patient_data_dirpath=self.base_patient_data_dirpath,
            beam_data_dirpath=self.beam_data_dirpath,
            energy=self.energy,
            dtd_load_dir=self.dtd_load_dir,
            dtd_scale_factor=self.dtd_scale_factor,
            reformat_along_beam=self.reformat_along_beam,
            batch_size=self.batch_size,
            crop_size=self.matrix_size,
            num_subarc_points=self.num_raytracing_subarc_points,
            logging_level=self.raydose_logging_level,
            arc=self.arc,
            control_points=self.control_points,
            average_raydoses=self.average_raydoses
        )

        self.patient = patient 
        self.arcs = self.raytracer.arcs
        self.all_arcs = self.raytracer.all_arcs
        self.logger.info((f'PATIENT: {patient} | ' 
                          f'ARCS: {[str(s) for s in self.arcs]} | '
                          f'CONTROL POINTS: {self.raytracer.full_control_point_count}'
                        ))
    
    def get_results(self):
        return self.arc_doses_dict, self.dicom_objects_dict, self.arc_runtimes_dict

    def run(self, patient=None):
        global_starttime = perf_counter()

        patient = patient if patient is not None else self.patients[0]
        self.initialize_for_patient(patient)
        self.arc_runtimes_dict['data_loading_time'] = round(self.raytracer.timings['load_and_set_global_data'], 2)

        for arc in self.arcs:
            self.logger.info(f'Running dose calculation for arc {arc}...')
            arc_starttime = perf_counter()

            self.raytracer.initialize_control_point_variables(arc=arc, control_points=self.control_points)
            arc_cp_indices = self.raytracer.control_points
            cp_batches = split_into_batches(arc_cp_indices, self.batch_size)

            # Calculate dose for first batch and initialize full arc dose to batch dose
            pred_batch_dose, mc_batch_dose = self.run_for_batch(cp_batches[0])
            pred_arc_dose = pred_batch_dose
            mc_arc_dose = mc_batch_dose if self.include_ground_truth else None

            # Calculate and add to full arc dose for all subsequent batches
            for i, batch in enumerate(cp_batches[1:]):
                pred_batch_dose, mc_batch_dose = self.run_for_batch(batch)
                
                if self.include_ground_truth:
                    if mc_batch_dose is None:
                        mc_batch_dose = cuda.device_array_like(pred_batch_dose)
                        mc_batch_dose[:] = 0

                pred_arc_dose = sum_cuda_arrays([pred_batch_dose, pred_arc_dose])
                mc_arc_dose = sum_cuda_arrays([mc_arc_dose, mc_batch_dose]) if (mc_arc_dose is not None and mc_batch_dose is not None) else mc_arc_dose

            # Tranfer predicted doses to CPU and append arc doses to results
            pred_arc_dose = self.raytracer.to_host(pred_arc_dose)
            pred_arc_dose = pred_arc_dose * self.raytracer.body_mask if self.return_to_original_geometry else pred_arc_dose
            self.arc_doses_dict[f'predicted_arc{arc}'] = pred_arc_dose
            if self.include_ground_truth and mc_arc_dose is not None:
                mc_arc_dose = mc_arc_dose * self.raytracer.body_mask if self.return_to_original_geometry else mc_arc_dose
                self.arc_doses_dict[f'ground_truth_arc{arc}'] = mc_arc_dose

            arc_runtime = round((perf_counter() - arc_starttime), 1)
            self.arc_runtimes_dict[f'arc{arc}'] = arc_runtime
            self.logger.info(f'Arc {arc} done. Runtime: {arc_runtime} s')

        # Sum predicted arc dose matrices to get predicted full plan dose
        pred_arc_doses = [v for k,v in self.arc_doses_dict.items() if 'predicted' in k]
        pred_full_plan_dose = np.sum(pred_arc_doses, axis=0)
        pred_full_plan_dose = pred_full_plan_dose * self.raytracer.body_mask if self.return_to_original_geometry else pred_full_plan_dose
        self.arc_doses_dict['predicted_full_plan_dose'] = pred_full_plan_dose
        
        # Create new DICOM RT dose file using predicted dose and include in dict
        if self.return_to_original_geometry:
            original_dicom_dose = deepcopy(self.raytracer.rt_dose)
            predicted_rt_dose = substitute_dose_array_in_dicom_dose(original_dicom_dose, pred_full_plan_dose)
            self.dicom_objects_dict['dicom_predicted_full_dose'] = predicted_rt_dose

        # Add Monte Carlo doses to results
        if self.include_ground_truth:
            mc_arc_doses = [v for k,v in self.arc_doses_dict.items() if 'ground_truth' in k]
            mc_full_plan_dose = np.sum(mc_arc_doses, axis=0)
            mc_full_plan_dose = mc_full_plan_dose * self.raytracer.body_mask if self.return_to_original_geometry else mc_full_plan_dose
            self.arc_doses_dict['ground_truth_full_plan_dose'] = mc_full_plan_dose

        # Add Eclipse dose to results
        if self.include_rtplan_dose:
            rt_plan_dose_array = get_dose_array_from_dicom_dose(self.raytracer.rt_dose)
            rt_plan_dose_array = rt_plan_dose_array * self.raytracer.body_mask if self.return_to_original_geometry else rt_plan_dose_array
            self.arc_doses_dict['rtplan_full_plan_dose'] = rt_plan_dose_array

        if self.save_dirpath is not None:
            arc_save_dirpath = join(self.save_dirpath, 'predicted_dose', self.model_architechture_name)
            dicom_save_dirpath = join(self.save_dirpath, 'dicom_objects', self.model_architechture_name)
            os.makedirs(arc_save_dirpath, exist_ok=True)
            os.makedirs(dicom_save_dirpath, exist_ok=True)

            arc_save_filename = f'{self.patient}_arc_dose_predictions_{os.path.basename(os.path.dirname(self.model_weights_filepath))}.npz'
            arc_save_filepath = os.path.join(arc_save_dirpath, arc_save_filename)
            np.savez_compressed(arc_save_filepath, **self.arc_doses_dict)

            for key, obj in self.dicom_objects_dict.items():
                dicom_save_filename = f'{self.patient}_{key}_{os.path.basename(os.path.dirname(self.model_weights_filepath))}.dcm'
                dicom_save_filepath = os.path.join(dicom_save_dirpath, dicom_save_filename)
                obj.save_as(dicom_save_filepath)

        global_runtime = round((perf_counter() - global_starttime), 1)
        self.arc_runtimes_dict['full_plan_time'] = global_runtime

        self.clear_gpu_memory()
        self.logger.info(f'DOSE CALCULATION COMPLETE. Full calculation time: {global_runtime} s')

    def run_for_arc_generator(self, arc, patient=None):
        patient = patient if patient is not None else self.patients[0]
        self.initialize_for_patient(patient)

        self.raytracer.initialize_control_point_variables(arc=arc, control_points=self.control_points)
        for cp in self.raytracer.control_points:
            pred_dose, mcdose = self.run_for_batch([cp])
            yield cp, pred_dose, mcdose

    def run_for_batch(self, batch):
        batch_size = len(batch)

        # Ray-trace input dose
        batch_vols, batch_rays, batch_mcdoses = self.raytracer.run_for_batch(batch)

        # Run deep learning input preparation, inference, and postprocessing
        batch_pred_doses = self.model.preprocess_infer_postprocess(batch_vols, batch_rays)

        # Check if all ground truth arrays are available and if they are to be processed
        process_ground_truths = batch_mcdoses is not None and self.include_ground_truth
        
        # Uncrop and orient along usual axes again (axial-coronal-sagittal)
        if self.return_to_original_geometry:
            arrays = [batch_pred_doses, batch_mcdoses] if process_ground_truths else [batch_pred_doses]
            arrays = self.raytracer.return_to_original_geometry_batched(arrays)

            batch_pred_doses = arrays[0]
            batch_mcdoses = arrays[1] if process_ground_truths else batch_mcdoses
        
        # Convert doses from cGy/MU to Gy
        if self.convert_GyperMU_to_Gy:
            batch_pred_doses, batch_mcdoses = self.raytracer.perform_GyperMU_to_Gy_conversion_gpu(
                batch, batch_pred_doses, batch_mcdoses, self.raytracer.acp_config['CMU'], process_ground_truths
            )       
         
        # Add invididual doses in batch to get predicted batch dose
        sum_dimensions = list(range(batch_size)) # Sum only over real batch_size dimensions (ignore padded dimensions)
        batch_pred_dose = sum_cuda_arrays(batch_pred_doses, index_list=sum_dimensions)
        batch_mcdose = sum_cuda_arrays(batch_mcdoses, index_list=sum_dimensions) if process_ground_truths else None

        if batch_mcdose is None and self.include_ground_truth:
            self.logger.warning(f'MC dose for batch {batch} is None.')

        if self.individual_cp_save_dir is not None:
            for cpname, vol, ray, pred, truth in zip(self.raytracer.cp_name, batch_vols, batch_rays, batch_pred_doses, batch_mcdoses):
                np.savez_compressed(
                    os.path.join(self.individual_cp_save_dir, f'{cpname}_STD.npz'),
                    vol=vol,
                    ray=ray,
                    pred=pred,
                    truth=truth
                )
                self.logger.info(f'Saved prediction for {cpname}.')

        for key in self.raytracer.timings:
            self.operation_times[key].append(self.raytracer.timings[key])
        for key in self.model.timings:
            self.operation_times[key].append(self.model.timings[key])
        
        return batch_pred_dose, batch_mcdose
    
    def clear_gpu_memory(self):
        gc.collect()
        cuda.current_context().deallocations.clear()
        cuda.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

    def set_logger(self, logging_level=logging.ERROR):
        logger_obj = logging.getLogger('PGINN')
        if logger_obj.hasHandlers():
            logger_obj.handlers.clear()
        logger_obj.setLevel(logging_level)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging_level) 
        formatter = logging.Formatter("%(name)s %(levelname)s:\t %(message)s")
        console_handler.setFormatter(formatter)
        logger_obj.addHandler(console_handler)
        self.logger = logger_obj
    
    def give_warnings(self):
        condition_message_pairs = [
            (not self.convert_GyperMU_to_Gy , 'Conversion from cGy/MU to Gy is turned off.'),
            (not self.return_to_original_geometry , 'Array reformatting back to original geometry after ray-tracing is turned off.'),
            (not self.include_ground_truth , 'Ground-truth dose loading is turned off. To turn on, provide dtd_load_dir.'),
            (self.save_dirpath is None , 'Results will not be saved to disk. To save results, provide save_dirpath.'),
            (self.individual_cp_save_dir is not None , 'Individual control point results will be saved. This may significantly increase runtime.')
        ]
        for condition, message in condition_message_pairs:
            if condition:
                self.logger.warning(message)

    def __str__(self):
        return (
            f"{self.patient.upper()}\n"
            f"Model name: {self.model_architechture_name}\n"
            f"Model weights path: {self.model_weights_filepath}\n"
            f"Batch size: {self.batch_size}"
        )
    
    def __len__(self):
        return self.raytracer.full_control_point_count
