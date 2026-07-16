from copy import deepcopy
import gc
import logging
import os
import sys
import math
import h5py
import pydicom
import numpy as np
import SimpleITK as sitk

from numba import cuda
import cupy as cp
_CUDA_AVAILABLE = cuda.is_available()
_CUDA_DEVICE_NAME = cuda.get_current_device()

from PGINN.raydose.dicom_processor import (
    load_ct_as_sitk,
    get_structure_number_from_name,
    generate_structure_mask_fast,
    extract_cp_configuration,
    get_geometry_from_dicom_dose,
)
from PGINN.raydose.geometry_utils import (
    resample_ct_to_new_geometry,
    crop_or_pad_batched_matrices,
    uncrop_matrices_gpu_batched,
    physical_to_index
)
from PGINN.raydose.gpu_raydose_calculator import compute_aperture_dose_batched
from PGINN.raydose.ssd_calculator import calculate_ssd_gpu_batched
from PGINN.raydose.ct_image_processor import map_hu_to_density
from PGINN.raydose.oblique_reformat import oblique_reformat_batched

from PGINN.utils.cuda_array_operations import (
    is_on_gpu, to_gpu,
    average_3D_numba_arrays,
    tile_volume_to_batch,
    scale_kernel_batched
)
from PGINN.utils.timing import timed_method
from PGINN.utils.general_functions import split_into_batches

class RaydoseComposer:

    TIMED_METHODS = {
        "load_and_set_global_data",
        "compute_batch_raydoses",
        "reformat_geometry_along_beam",
        "crop_or_pad",
        "return_to_original_geometry_batched",
        "perform_GyperMU_to_Gy_conversion_gpu"
    }

    def __init__(
            self,
            patient,
            base_patient_data_dirpath,
            beam_data_dirpath,
            energy,
            arc=None,
            control_points=None,
            **kwargs
            ):
        """
        Parameters
        ----------
        patient : str 
            Name of patient folder to locate in base_patient_data_dirpath

        base_patient_data_dirpath : str or pathlib.Path
            Path to the directory containing patient folders, which must contain CT, RP, RS, RD files

        beam_data_dirpath : str or pathlib.Path
            Path to the directory containing saved beam PDD and OAR numpy arrays

        energy : str or float
            Beam energy index for PDD and OAR arrays

        arc : int, optional
            Optional arc label to compute. All arcs computed if not provided.

        control_points : list, optional
            Optional control point labels to compute. All arcs computed if not provided.
        """
        self.timings = {}

        self.load_and_set_global_data(
            patient,
            base_patient_data_dirpath,
            beam_data_dirpath,
            energy,
            **kwargs
        )

        self.initialize_control_point_variables(
            arc,
            control_points
        )

        self.logger.info('Ray-tracing pipeline initialized.')

    def load_and_set_global_data(self,
                        patient,
                        base_patient_data_dirpath,
                        beam_data_dirpath,
                        energy=1,
                        acp_config_path=None,
                        batch_size=1,
                        crop_arrays=True,
                        reformat_along_beam=True,
                        crop_size=(96, 96, 64),
                        use_GPU=True,
                        average_raydoses=True,
                        num_subarc_points=2,
                        dtd_load_dir=None,
                        dtd_scale_factor=1.00,
                        randomize_geometry=False,
                        std_save_path=None,
                        logging_level=logging.INFO,
                        run_legacy_pipeline=False,
                        ):
        """
        Parameters
        ----------
        patient : str 
            Name of patient folder to locate in base_patient_data_dirpath

        base_patient_data_dirpath : str or pathlib.Path
            Path to the directory containing patient folders, which must contain CT, RP, RS, RD files

        beam_data_dirpath : str or pathlib.Path
            Path to the directory containing saved beam PDD and OAR numpy arrays

        energy : str or float
            Beam energy index for PDD and OAR arrays

        acp_config_path : str or pathlib.Path, optional
            Path to acp_config data that has been previously extracted and saved as a numpy file
            If provided, the loaded data will be used. Otherwise, the data will be extracted automatically from the RT Plan DICOM

        batch_size : int
            Number of sub-arcs (i.e. control points) to process in parallel

        crop_arrays : bool
            Whether to crop ray-traced doses about the isocenter
        
        reformat_along_beam : bool
            Whether to resample arrays to a beam's eye view coordinate system before cropping
        
        crop_size : tuple
            Matrix size of cropped arrays (only used if crop_arrays is True)

        use_GPU: bool
            Set false to use CPU version

        average_raydoses: bool
            Whether to return the average of raydoses calculated at points along sub-arc.
            If false, will return all raydoses for all points as part of the output array
            (i.e. output will have a 'raydose point channel')

        num_subarc_points: int
            Number of points along sub-arc for which to compute raydoses.
            Default of 2 calculates raydoses at both control points of sub-arc.
            Setting to 1 calculates raydoses at second (end) control point of sub-arc.
            Setting > 2 calculates raydoses at both control points plus at interpolated points between them. 

        dtd_load_dir : str or pathlib.Path, optional
            Path to the directory containing Monte Carlo dose npz files ("_DTD files").
            If not provided, Monte Carlo doses will not be loaded.
        
        dtd_scale_factor : float
            Constant scale factor applied to MC dose arrays

        randomize_geometry : bool
            Whether to change isocenter and collimator angles of control points.

        std_save_path : str or pathlib.Path, optional
            Path to where training data "_STD files" will be saved.
            If is a directory, arrays for each sub-arc will be saved in an individual npz file.
            If is path to an hf5 file, all arrays will be appended to this one file.
        """

        # Set ray-tracing settings
        self.set_logger(logging_level)
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.use_GPU = use_GPU and _CUDA_AVAILABLE
        self.reformat_along_beam = reformat_along_beam
        self.crop_arrays = crop_arrays
        self.run_legacy_pipeline = run_legacy_pipeline
        self.average_raydoses = average_raydoses
        self.num_subarc_points = num_subarc_points
        self.output_raydoses = 1 if self.average_raydoses else self.num_subarc_points
        self.dtd_scale_factor = dtd_scale_factor

        # Set required data
        self.patient = patient
        self.energy = 1 if energy == 0 else energy
        self.set_paths(
            base_patient_data_dirpath, 
            beam_data_dirpath,
            dtd_load_dir,
            std_save_path,
        )

        # Load patient and beam files.
        self.ct_sitk = load_ct_as_sitk(self.ct_filepath)
        self.rt_dose = pydicom.dcmread(self.rd_filepath)
        self.rt_plan = pydicom.dcmread(self.rp_filepath)
        self.rt_struct = pydicom.dcmread(self.rs_filepath)
        self.pdd_list = np.load(self.pdd_filepath).astype(float) 
        self.oar_list = np.load(self.oar_filepath).astype(float)

        # Get patient binary body mask from RT Struct
        self.body_struct_num = get_structure_number_from_name(self.rt_struct, 'BODY')
        self.body_mask = generate_structure_mask_fast(self.rt_dose, self.rt_struct, self.body_struct_num).astype(np.int8)
        self.body_mask_shape = self.body_mask.shape
        self.gpu_body_mask = self.to_device(self.body_mask.astype(np.int8))
        self.n_voxels = int(np.prod(self.body_mask_shape))

        # Get dose geometry from RT Dose
        origin, spacing, shape, dirs = get_geometry_from_dicom_dose(self.rt_dose)
        self.dose_origin = origin
        self.dose_spacing = spacing
        self.dose_shape = shape
        self.dose_direction_matrix = dirs

        # Resample CT image to the dose geometry
        houndsfield_array = resample_ct_to_new_geometry(
            self.ct_sitk, self.dose_origin, self.dose_spacing, self.dose_shape, self.dose_direction_matrix,
            return_as_array=True
        )

        # Map CT values in HU to physical density values
        relative_electron_density_array, physical_density_array = map_hu_to_density(houndsfield_array)
        
        # Send physical density array to GPU if available
        self.pdensity_array = (physical_density_array * self.body_mask).astype(np.float32)
        self.red_density_array = (relative_electron_density_array * self.body_mask).astype(np.float32)
        self.pdensity_array = self.to_device(self.pdensity_array)
        self.red_density_array = self.to_device(self.red_density_array)

        # Set MLC pairs and max boundary
        self.set_mlc_data()

        # Extract control point info struct from RT Plan
        self.load_full_acp_config_data(
            acp_config_path=acp_config_path,
            randomize_geometry=randomize_geometry
        )
        
        # Get idx for beam to get correct PDD and OAR data (e.g. 6 MV beam has energy_idx = 1)
        self.energy_idx = self.full_acp_config[0]['col_idx']
        self.energy_idx = 1 if self.energy_idx == 0 else self.energy_idx

        # Set beam data (pdd, oar)
        self.pdd_depths = self.to_device(self.pdd_list[:, 0].astype(np.float32))
        self.pdd_values = self.to_device(self.pdd_list[:, self.energy_idx].astype(np.float32))
        self.oar_dist = self.to_device(self.oar_list[:,0].astype(np.float32))
        self.oar_values = self.to_device(self.oar_list[:, self.energy_idx].astype(np.float32))

        # Calibraton depth by energy (dmax) and max PDD depth
        ref_depth = [                                                                   
            self.pdd_list[69, 0], 
            self.pdd_list[69, 0], 
            self.pdd_list[119, 0], 
            self.pdd_list[109, 0]
        ]
        self.ref_depth = ref_depth[self.energy_idx - 1]
        self.max_pdd_depth = np.max(self.pdd_depths)
        self.ssd_ref = 1000.0 # mm

        # Store pipeline data
        self.full_control_point_count = len(self.full_acp_config)
        self.all_arcs = sorted(list(np.unique(self.full_acp_config['arc_name'].astype('str'))))

        # Pre-allocate aperture doses array on gpu to fill during ray-tracing
        self.aperture_doses = cuda.device_array((self.batch_size, self.num_subarc_points, self.n_voxels), dtype=np.float32)

        self.logger.info('Patient and beam data loaded.')

    def initialize_control_point_variables(self, arc=None, control_points=None):
        self.set_acp_config_data(arc=arc, control_points=control_points)
        self.clear_batch_data_list()

    def clear_batch_data_list(self):
        # Initialize current control point batch info
        self.cp = []
        self.cp_arc = []
        self.cp_idx = []
        self.cp_name = []

        # Initialize data for starting control points of sub-arcs
        self.isocenter_a = []
        self.zeroed_isocenter_a = []
        self.collimator_angle_a = []
        self.gantry_angle_a = []

        # Initialize data for ending control points of sub-arcs
        self.isocenter_b = []
        self.zeroed_isocenter_b = []
        self.collimator_angle_b = []
        self.gantry_angle_b = []

        self.original_geometry_dict = []
        self.beam_direction_vector = []
        self.reoriented_geometry_dict = []
        self.crop_metadata_dict = []

    def set_current_batch_info(self, batch):
        self.clear_batch_data_list()
        self.cp = batch

        # If present, remove first control point of each arc (sub-arcs are indexed by the second control point)
        for cp in batch:
            if self.acp_config['cp_idx'][cp] == '0':
                batch.remove(cp)

        row_a = self.acp_config[batch[0]-1]
        
        for cp in batch:
            row_b = self.acp_config[cp]

            cp_arc = row_b['arc_name'].astype('U20')
            cp_idx = row_b['cp_idx'].astype('U20')
            cp_name = f'{self.patient}_arc{cp_arc}_CP{cp_idx}'

            self.cp_arc.append(cp_arc)
            self.cp_idx.append(cp_idx)
            self.cp_name.append(cp_name)
            self.isocenter_a.append(row_a['isocenter'])
            self.isocenter_b.append(row_b['isocenter'])
            self.zeroed_isocenter_a.append(self.dose_origin - row_a['isocenter'])
            self.zeroed_isocenter_b.append(self.dose_origin - row_b['isocenter'])
            self.collimator_angle_a.append(row_a['collimator_angle'])
            self.collimator_angle_b.append(row_b['collimator_angle'])
            self.gantry_angle_a.append(row_a['gantry_angle'])
            self.gantry_angle_b.append(row_b['gantry_angle'])

            angle_to_use = row_b['gantry_angle']
            ang_rad = np.radians(angle_to_use)
            self.beam_direction_vector.append(-np.array([np.sin(ang_rad),  -np.cos(ang_rad),  0]))
        
            self.original_geometry_dict.append({
                'origin' : self.dose_origin,
                'spacing' : self.dose_spacing,
                'size' : self.dose_shape,
                'direction_matrix' : self.dose_direction_matrix,
                'isocenter' : row_a['isocenter'],
                'isocenter_index' : physical_to_index(
                                        row_a['isocenter'], self.dose_origin, 
                                        self.dose_spacing, self.dose_direction_matrix
                                    )
            })

            row_a = row_b
    
        self.vol = []
        self.ray = []
        self.mcdose = []
        self.last_cp_endpoint_dose = None

    def load_full_acp_config_data(self, acp_config_path=None, randomize_geometry=False):
        if acp_config_path is not None and os.path.exists(acp_config_path):
            self.full_acp_config = np.load(acp_config_path)['ACPConfig']
        else:
            self.full_acp_config = extract_cp_configuration(
                    self.rt_plan, self.energy, self.std_save_path, self.patient,
                    randomize_geometry=randomize_geometry,
                    save_result=False
                )

    def set_acp_config_data(self, arc=None, control_points=None):
        self.acp_config = deepcopy(self.full_acp_config)
        
        if arc is not None:
            self.acp_config = self.acp_config[self.acp_config['arc_name'].astype('U20') == str(arc)]

        self.arcs = list(np.unique(self.acp_config['arc_name'].astype('str')))
        self.control_points = control_points or list(range(1, len(self.acp_config)))
        self.control_point_count = len(self.control_points)
        self.batches = split_into_batches(self.control_points, self.batch_size)

        # If geometry was randomized, isocenter and collimator angle will vary from cp to cp
        self.geometry_not_randomized = (
                (tuple(self.acp_config[0]['isocenter']) == tuple(self.acp_config[1]['isocenter'])) and
                (self.acp_config[0]['collimator_angle'] == self.acp_config[1]['collimator_angle'])
            )

        # Generate and send necessary data to GPU for calculation
        self.generate_gpu_interpolated_acp_data()

    def generate_gpu_interpolated_acp_data(self):
        acp_control_points = list(range(1, len(self.acp_config)))
        m = len(acp_control_points)
        n_points = self.num_subarc_points
        mlc_pairs = self.mlc_pairs

        t_values = np.linspace(0, 1, n_points, dtype=np.float32)

        dtype = np.dtype([
            ("sx", "f4"), ("sy", "f4"), ("sz", "f4"), ("source_norm", "f4"),
            ("cg", "f4"), ("sg", "f4"), ("ca", "f4"), ("sa", "f4"),
            ("jx_min", "f4"), ("jx_max", "f4"),
            ("jy_min", "f4"), ("jy_max", "f4"),
            ("ox", "f4"), ("oy", "f4"), ("oz", "f4"),
        ])

        params = np.zeros((m, n_points), dtype=dtype)
        mlc_x_block = np.zeros((m, n_points, 2, mlc_pairs), dtype=np.float32)
        mlc_z_block = np.zeros((m, n_points, mlc_pairs + 1), dtype=np.float32)

        cfg_start = self.acp_config[:-1]
        cfg_end   = self.acp_config[1:]

        ga_s = cfg_start['gantry_angle'].astype(np.float32)
        ga_e = cfg_end['gantry_angle'].astype(np.float32)
        delta_mid = (ga_e - ga_s + 180) % 360 - 180
        gantry_mid = (ga_s + 0.5 * delta_mid) % 360

        # Isocenter and collimator angle obtained from the first control point.
        # Generally isocenter won't change between points so this won't matter, 
        # but it does in the case of our training data with randomized geometry. 
        # Our MC simulations of sub-arcs were performed with isocenter and collimator angle of first control point.
        iso_ipps_static = (self.dose_origin - cfg_start['isocenter']).astype(np.float32)
        coll_angles = cfg_start['collimator_angle'].astype(np.float32)[:, None]

        ssd = calculate_ssd_gpu_batched(
            gantry_mid, iso_ipps_static, self.dose_spacing, self.gpu_body_mask,
            threshold=np.float32(0.3),
        )

        if n_points == 1:
            gantry_angles = cfg_end['gantry_angle'].astype(np.float32)[:, None]
            
            ox_arr = iso_ipps_static[:, 0:1]
            oy_arr = iso_ipps_static[:, 1:2]
            oz_arr = iso_ipps_static[:, 2:3]

            xj_b = cfg_end['x_jaw'].astype(np.float32)
            yj_b = cfg_end['y_jaw'].astype(np.float32)

            jaw_xmins = xj_b[:, 0:1]
            jaw_xmaxs = xj_b[:, 1:2]
            jaw_ymins = yj_b[:, 0:1]
            jaw_ymaxs = yj_b[:, 1:2]

            x_mlc_b = cfg_end['x_mlc'].astype(np.float32)
            left_b  = x_mlc_b[:, :mlc_pairs]
            right_b = x_mlc_b[:, mlc_pairs:]

            mlc_left_xp  = left_b[:, None, :]
            mlc_right_xp = right_b[:, None, :]

            z_mlc_b = cfg_end['z_mlc'].astype(np.float32)
            mlc_zp = z_mlc_b[:, None, :]
        
        else:
            t = t_values[None, :]

            ga_start = cfg_start['gantry_angle'].astype(np.float32)[:, None]
            ga_end   = cfg_end['gantry_angle'].astype(np.float32)[:, None]
            delta = (ga_end - ga_start + 180) % 360 - 180
            gantry_angles = ga_start + t * delta

            # Lock collimator and isocenter to start control point across all t
            coll_angles = np.broadcast_to(coll_angles, (m, n_points))

            ox_arr = np.broadcast_to(iso_ipps_static[:, 0:1], (m, n_points))
            oy_arr = np.broadcast_to(iso_ipps_static[:, 1:2], (m, n_points))
            oz_arr = np.broadcast_to(iso_ipps_static[:, 2:3], (m, n_points))

            xj_a = cfg_start['x_jaw'].astype(np.float32)
            xj_b = cfg_end['x_jaw'].astype(np.float32)
            yj_a = cfg_start['y_jaw'].astype(np.float32)
            yj_b = cfg_end['y_jaw'].astype(np.float32)

            jaw_xmins = xj_a[:, 0:1] + t * (xj_b[:, 0:1] - xj_a[:, 0:1])
            jaw_xmaxs = xj_a[:, 1:2] + t * (xj_b[:, 1:2] - xj_a[:, 1:2])
            jaw_ymins = yj_a[:, 0:1] + t * (yj_b[:, 0:1] - yj_a[:, 0:1])
            jaw_ymaxs = yj_a[:, 1:2] + t * (yj_b[:, 1:2] - yj_a[:, 1:2])

            x_mlc_a = cfg_start['x_mlc'].astype(np.float32)
            x_mlc_b = cfg_end['x_mlc'].astype(np.float32)

            left_a  = x_mlc_a[:, :mlc_pairs]
            left_b  = x_mlc_b[:, :mlc_pairs]
            right_a = x_mlc_a[:, mlc_pairs:]
            right_b = x_mlc_b[:, mlc_pairs:]

            mlc_left_xp  = left_a[:, None, :]  + t[..., None] * (left_b[:, None, :]  - left_a[:, None, :])
            mlc_right_xp = right_a[:, None, :] + t[..., None] * (right_b[:, None, :] - right_a[:, None, :])

            z_mlc_a = cfg_start['z_mlc'].astype(np.float32)
            z_mlc_b = cfg_end['z_mlc'].astype(np.float32)
            mlc_zp = z_mlc_a[:, None, :] + t[..., None] * (z_mlc_b[:, None, :] - z_mlc_a[:, None, :])

        ang_rad = np.radians(gantry_angles, dtype=np.float32)
        sin_g = np.sin(ang_rad)
        cos_g = np.cos(ang_rad)

        sx_arr = 1000.0 * sin_g
        sy_arr = -1000.0 * cos_g
        sz_arr = np.zeros_like(sx_arr, dtype=np.float32)
        source_norm_arr = np.full_like(sx_arr, 1000.0, dtype=np.float32)

        gantry_rot = np.radians(360.0 - gantry_angles, dtype=np.float32)
        cos_gantry_arr = np.cos(gantry_rot)
        sin_gantry_arr = np.sin(gantry_rot)

        coll_rot = np.radians(360.0 - coll_angles, dtype=np.float32)
        cos_ca_arr = np.cos(coll_rot)
        sin_ca_arr = np.sin(coll_rot)

        params["sx"][:]          = sx_arr
        params["sy"][:]          = sy_arr
        params["sz"][:]          = sz_arr
        params["source_norm"][:] = source_norm_arr
        params["cg"][:]          = cos_gantry_arr
        params["sg"][:]          = sin_gantry_arr
        params["ca"][:]          = cos_ca_arr
        params["sa"][:]          = sin_ca_arr
        params["jx_min"][:]      = jaw_xmins
        params["jx_max"][:]      = jaw_xmaxs
        params["jy_min"][:]      = jaw_ymins
        params["jy_max"][:]      = jaw_ymaxs

        mlc_x_block[:, :, 0, :] = mlc_left_xp
        mlc_x_block[:, :, 1, :] = mlc_right_xp
        mlc_z_block[:, :, :]    = mlc_zp

        params["ox"][:] = ox_arr
        params["oy"][:] = oy_arr
        params["oz"][:] = oz_arr

        self.raytrace_calc_params = self.to_device(params)
        self.ssd = ssd
        self.mlc_x_block = self.to_device(mlc_x_block)
        self.mlc_z_block = self.to_device(mlc_z_block)

    def set_mlc_data(self):
        self.mlc_pairs = np.int16(60)
        self.max_leaf_boundary = np.float32(110.0)
        
        # Turned off for now.
        # try:
        #     pairs = []
        #     max_boundary = []
        #     for beam in self.rt_plan.BeamSequence:
        #         if beam.TreatmentDeliveryType != 'TREATMENT':
        #             continue
        #         for device in beam.BeamLimitingDeviceSequence:
        #             pairs.append(device.NumberOfLeafJawPairs)
        #             if hasattr(device, "LeafPositionBoundaries"):
        #                 max_boundary.append(np.max(device.LeafPositionBoundaries))
        #     self.mlc_pairs = np.int16(np.max(pairs))
        #     self.max_leaf_boundary = np.float32(np.max(max_boundary))
        # except Exception as e:
        #     # Default to hard-coded current L2 MLC values
        #     self.mlc_pairs = np.int16(60)
        #     self.max_leaf_boundary = np.float32(110.0)

    def run(self):
        if self.run_legacy_pipeline and not self.use_GPU:
            self.run_legacy()
        else:
            for batch in self.batches:
                self.run_for_batch(batch)

    def run_generator(self):
        for batch in self.batches:
            vol, ray, mcdose = self.run_for_batch(batch)
            yield batch, vol, ray, mcdose

    def run_for_arc(self, arc):
        self.initialize_control_point_variables(arc=arc)
        self.run()
    
    def run_for_arc_generator(self, arc):
        self.initialize_control_point_variables(arc=arc)
        for batch in self.batches:
            vol, ray, mcdose = self.run_for_batch(batch)
            yield batch, vol, ray, mcdose
            
    def run_for_cp(self, cp):
        return self.run_for_batch([cp])

    def run_for_batch(self, batch):
        self.set_current_batch_info(batch)
        batch_size = len(batch)
        
        # Define (batch_size, nx, ny, nz) cuda gpu arrays | (nx, ny, nz) = dose_shape
        self.vol = tile_volume_to_batch(self.pdensity_array, batch_size)
        self.ray = self.compute_batch_raydoses()
        self.mcdose = self.load_batch_monte_carlo_doses(batch)

        # Perform reformatting and/or cropping on batch, as appropriate
        arrays = [self.vol, self.ray, self.mcdose] if self.mcdose is not None else [self.vol, self.ray]
        arrays = self.reformat_geometry_along_beam(arrays) if self.reformat_along_beam else arrays
        arrays = self.crop_or_pad(arrays) if self.crop_arrays else arrays
        self.vol, self.ray, self.mcdose = [*arrays, None][:3]

        save_log_message = ''
        if self.std_save_path is not None:
            self.save_std_result_for_batch()
            save_log_message = ' STD files saved.'

        for i, cp in enumerate(batch):
            self.logger.info(f'Ray-tracing done for: {self.cp_name[i]}.{save_log_message}')

        return self.vol, self.ray, self.mcdose

    def compute_batch_raydoses(self):
        # Compute aperture doses for the num_subarc_points of the current sub-arc
        raydoses = compute_aperture_dose_batched(
            cp_list=self.cp,
            n_voxels=self.n_voxels,
            body_mask=self.gpu_body_mask,
            body_mask_shape=self.body_mask_shape,
            dose_grid=self.dose_spacing,
            density_array=self.red_density_array,
            ssd_ref=self.ssd_ref, ref_depth=self.ref_depth, max_pdd_depth=self.max_pdd_depth,
            pdd_depths=self.pdd_depths, pdd_values=self.pdd_values, oar_dist=self.oar_dist, oar_values=self.oar_values,
            case_params=self.raytrace_calc_params,
            ssd_arr=self.ssd,
            mlc_x_block=self.mlc_x_block,
            mlc_z_block=self.mlc_z_block,
            points_per_subarc=self.num_subarc_points,
            num_mlc_pairs=self.mlc_pairs, max_leaf_boundary=self.max_leaf_boundary,
            use_GPU=self.use_GPU, keep_arrays_in_GPU=True,
            reuse_first_subarc_point_dose=self.geometry_not_randomized,
            last_cp_endpoint_dose=self.last_cp_endpoint_dose,
            aperture_doses=self.aperture_doses
            )

        # Save second control point dose of last sub-arc in batch
        self.last_cp_endpoint_dose = raydoses[-1, -1, :, :, :]

        # No need to average if only one point was computed
        if self.num_subarc_points == 1:
            return raydoses

        # Compute control point raydose as the mean of the individual sub-arc aperture doses
        raydoses = self.average_function(raydoses) if self.average_raydoses else raydoses

        return raydoses

    def reformat_geometry_along_beam(self, arrays_to_reformat):
        # Use current gantry angle but isocenter and collimator angle to match previous control point
        geometry_dict = self.original_geometry_dict
        for i, d  in enumerate(geometry_dict):
            d['isocenter'] = self.isocenter_a[i]
            d['collimator_angle'] = self.collimator_angle_a[i]

        reformatted_arrays, new_geometry_dict = oblique_reformat_batched(
                                                    arrays_to_reformat,
                                                    geometry_dict,
                                                    self.beam_direction_vector,
                                                    match_size=False,
                                                    beam_axis=0,
                                                    use_GPU=self.use_GPU,
                                                    inverse=False
                                                    ) 
        self.reoriented_geometry_dict = new_geometry_dict
        return reformatted_arrays

    def crop_or_pad(self, arrays_to_crop):
        if self.reformat_along_beam and self.reoriented_geometry_dict is not None:
            isocenter_index = [d['isocenter_index'] for d in self.reoriented_geometry_dict]
        else:
            isocenter_index = [physical_to_index(
                                        iso, self.dose_origin, 
                                        self.dose_spacing,
                                        self.dose_direction_matrix
                                        )
                                for iso in self.isocenter_a]

        cropped_arrays, crop_metadata = crop_or_pad_batched_matrices(
                            arrays_to_crop,
                            isocenter_index,
                            crop_dim=self.crop_size
                            )
        self.crop_metadata_dict = crop_metadata
        return cropped_arrays

    def return_to_original_geometry_batched(self, arrays):
        # Uncrop matrices back to original shape by zero-padding
        arrays = uncrop_matrices_gpu_batched(arrays, self.crop_metadata_dict, pad_value=0)

        if self.reformat_along_beam:
            # Reformat matrices back to initial orientation
            initial_geometry_dict = deepcopy(self.original_geometry_dict)
            for i, d in enumerate(initial_geometry_dict):
                d['isocenter'] = self.isocenter_a[i]

            arrays, _ = oblique_reformat_batched(
                                        arrays,
                                        self.reoriented_geometry_dict,
                                        self.beam_direction_vector, # Note: Beam vector ignored on inverse computation
                                        match_size=False,
                                        beam_axis=0,
                                        use_GPU=self.use_GPU,
                                        inverse=True,
                                        initial_geometry=initial_geometry_dict,
                                    )            
        return arrays 

    def perform_GyperMU_to_Gy_conversion_gpu(
        self,
        batch,
        predicted_doses,
        batch_ground_truth_doses,
        cmu_list,
        process_ground_truths,
        cGy_to_Gy=True
    ):
        cGy_factor = 0.01 if cGy_to_Gy else 1.0

        cmu = cmu_list[batch] * cGy_factor
        cmu_arr = np.asarray(cmu, dtype=np.float32)
        cmu_dev = cuda.to_device(cmu_arr)

        batch_size, nx, ny, nz = predicted_doses.shape

        threads = (8, 8, 8)
        blocks = (
            math.ceil(nx / threads[0]),
            math.ceil(ny / threads[1]),
            math.ceil(nz / threads[2]),
        )

        # Scale predicted doses in one kernel launch
        scale_kernel_batched[blocks, threads](predicted_doses, cmu_dev)

        # Optionally scale ground truth doses with the same factors
        if process_ground_truths and batch_ground_truth_doses is not None:
            # Reuse cmu_dev, same shape
            scale_kernel_batched[blocks, threads](batch_ground_truth_doses, cmu_dev)

        return predicted_doses, batch_ground_truth_doses

    def run_legacy(self):
        self.logger.info('Running legacy pipeline')
        from PGINN.raydose.legacy_pipeline import compose_aperture_inputs_legacy
        compose_aperture_inputs_legacy(
            self.acp_config, self.generate_proc_data_legacy(), self.pdensity_array
        )
        
    def to_device(self, x):
        if self.use_GPU:
            if not is_on_gpu(x):
                if np.issubdtype(x.dtype, np.floating):
                    return to_gpu(x.astype(np.float32))
                elif np.issubdtype(x.dtype, np.integer):
                    return to_gpu(x.astype(np.int8))
                else:
                    return to_gpu(x)
        return x

    def to_host(self, x):
        if self.use_GPU and is_on_gpu(x):
            return x.copy_to_host()
        return x

    def average_function(self, array_list):
        if self.use_GPU:
            return average_3D_numba_arrays(array_list)
        return np.mean(array_list)

    def set_logger(self, logging_level=logging.ERROR):
        logger_obj = logging.getLogger('Raydose')
        if logger_obj.hasHandlers():
            logger_obj.handlers.clear()
        logger_obj.setLevel(logging_level)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging_level) 
        formatter = logging.Formatter("%(name)s %(levelname)s:\t %(message)s")
        console_handler.setFormatter(formatter)
        logger_obj.addHandler(console_handler)
        self.logger = logger_obj
    
    def set_paths(self,
                  base_patient_data_dirpath,
                  beam_data_dirpath,
                  dtd_load_dir=None,
                  std_save_path=None
                  ):

        self.patient_data_dirpath = os.path.join(base_patient_data_dirpath, self.patient)
        self.beam_data_dirpath = beam_data_dirpath

        for field in ['patient_data_dirpath', 'beam_data_dirpath']:
            filepath = getattr(self, field)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f'{field}={filepath} folder not found.')

        patient_data = os.listdir(self.patient_data_dirpath)
        pt_text_dict = {
            'DICOM RTDOSE' : [f for f in patient_data if 'RD' in f and f.endswith('.dcm')],
            'DICOM RTPLAN' : [f for f in patient_data if 'RP' in f and f.endswith('.dcm')],
            'DICOM RTSTRUCT' : [f for f in patient_data if 'RS' in f and f.endswith('.dcm')],
            'CT (NRRD OR DICOM)' : [f for f in patient_data if 'CT' in f and (f.endswith('.nrrd') or f.endswith('.dcm'))],
        }
        for text, filelist in pt_text_dict.items():
            if len(filelist) == 0:
                raise FileNotFoundError(f'{text} file not found in patient data folder.')
        filenames = list(pt_text_dict.values())

        self.rd_filepath = os.path.join(self.patient_data_dirpath, filenames[0][0])
        self.rp_filepath = os.path.join(self.patient_data_dirpath, filenames[1][0])
        self.rs_filepath = os.path.join(self.patient_data_dirpath, filenames[2][0])

        # Set CT image path depending on whether NRRD image already exists or DICOM image series need to be collected
        nrrd_files = [f for f in filenames[3] if f.endswith('.nrrd')]
        self.ct_filepath = os.path.join(self.patient_data_dirpath, nrrd_files[0]) if len(nrrd_files) > 0 else self.patient_data_dirpath

        beam_data = os.listdir(self.beam_data_dirpath)
        beam_text_dict = {
            'PDD' : [f for f in beam_data if 'PDD' in f],
            'OAR' : [f for f in beam_data if 'OAR' in f],
        }
        for text, filelist in beam_text_dict.items():
            if len(filelist) == 0:
                raise FileNotFoundError(f'{text} file not found in beam data folder.')

        filenames = list(beam_text_dict.values())
        self.pdd_filepath = os.path.join(self.beam_data_dirpath, filenames[0][0])
        self.oar_filepath = os.path.join(self.beam_data_dirpath, filenames[1][0])

        # See documentation for explanation of rtd, dtd, and std files.
        # If path is not set, pipeline will not load/save files.
        self.dtd_load_dir = dtd_load_dir
        self.std_save_path = std_save_path
        self.std_hf5 = None

        if self.std_save_path is not None:
            if os.path.isdir(self.std_save_path):
                os.makedirs(self.std_save_path, exist_ok=True)
            if self.std_save_path.endswith('.hf5'):
                if not os.path.exists(self.std_save_path):
                    self.std_hf5 = h5py.File(std_save_path, "a") 
                    self.std_hf5.create_dataset("case_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"), compression=None, chunks=True)
                    self.std_hf5.create_dataset("vol",  shape=(0, *self.crop_size), maxshape=(None, *self.crop_size), chunks=(1, *self.crop_size), dtype=np.float32, compression=None)
                    self.std_hf5.create_dataset("dose", shape=(0, *self.crop_size), maxshape=(None, *self.crop_size), chunks=(1, *self.crop_size), dtype=np.float32, compression=None)

                    ray_shape = (0, self.output_raydoses, *self.crop_size)
                    ray_chunks = (1, self.output_raydoses, *self.crop_size)
                    ray_max_shape = (None, self.output_raydoses, *self.crop_size)
                    self.std_hf5.create_dataset("ray",  shape=ray_shape, maxshape=ray_max_shape, chunks=ray_chunks, dtype=np.float32, compression=None)
                else:
                    self.std_hf5 = h5py.File(self.std_save_path, "a")

    def save_std_result_for_batch(self):
        vols = self.to_host(self.vol)
        rays = self.to_host(self.ray)
        mcdoses = self.to_host(self.mcdose) if self.mcdose is not None else None
        
        for i, cp in enumerate(self.cp):
            cp_name = self.cp_name[i]
            vol = vols[i]
            ray = rays[i]
            mcdose = mcdoses[i] if mcdoses is not None else None

            if self.std_hf5 is not None:
                self.append_std_case(f'{cp_name}_STD', vol, ray, mcdose)
            else:
                if os.path.isdir(self.std_save_path):
                    save_filepath = os.path.join(self.std_save_path, f'{cp_name}_STD.npz')
                    save_dict = {
                        'vol' : vol,
                        'ray' : ray,
                    }
                    if mcdose is not None:
                        save_dict['dose'] = mcdose
                    np.savez_compressed(
                        save_filepath,
                        **save_dict
                    )

    def append_std_case(self, case_id, vol, ray, dose):
        dset = self.std_hf5["case_ids"]
        old = dset.shape[0]
        dset.resize(old + 1, axis=0)
        dset[old] = case_id

        dset = self.std_hf5["vol"]
        dset.resize(old + 1, axis=0)
        dset[old] = self.to_host(vol)

        dset = self.std_hf5["dose"]
        dset.resize(old + 1, axis=0)
        dset[old] = self.to_host(dose) if dose is not None else np.full(vol.shape, np.nan, dtype=np.float32)

        dset = self.std_hf5["ray"]
        dset.resize(old + 1, axis=0)
        dset[old] = self.to_host(ray)

    def load_batch_monte_carlo_doses(self, batch):
        mcdose = None
        if self.dtd_load_dir is not None:
            doses_list = []
            stack_doses = True
            for idx, cp in enumerate(batch):
                cp_name = self.cp_name[idx]
                mcdose_path = os.path.join(self.dtd_load_dir, self.patient, f'{cp_name}_DTD.npz')
                if not os.path.exists(mcdose_path):
                    self.logger.warning(f'Missing ground truth dose: {cp_name}_DTD.npz')
                    stack_doses = False
                    break
                else:
                    doses_list.append(self.dtd_scale_factor * np.load(mcdose_path)['dose'].astype(np.float32))
            if stack_doses:
                dose_batch = np.stack(doses_list, axis=0)
                mcdose = self.to_device(dose_batch)
        return mcdose

    # Generate proc_data dict used in previous versions of library
    def generate_proc_data_legacy(self):
        return {
            'pat_name' : self.patient,
            'save_dir' : self.std_save_path,
            'dtd_load_dir' : self.dtd_load_dir,
            'density_array' : self.to_host(self.red_density_array), 
            'body_mask' : self.body_mask,
            'dose_grid' : self.dose_spacing,
            'img_pos_pat' : self.dose_origin,
            'ref_depth' : self.ref_depth,
            'ssd_ref' : self.ssd_ref,
            'max_pdd_depth' : self.max_pdd_depth,
            'pdd_values' : self.pdd_values,
            'pdd_depths' : self.pdd_depths,
            'oar_dist' : self.oar_dist,
            'oar_values' : self.oar_values,
            'crop_vol' : self.crop_size,
            'crop' : self.crop_arrays,
            'control_points' : self.control_points
        }
    
    # Generate cp_config dict used in previous versions of library
    def generate_cp_configs_legacy(self, cp):
        cp_config_a = deepcopy(self.acp_config[cp - 1])
        cp_config_b = deepcopy(self.acp_config[cp])
        return cp_config_a, cp_config_b

    def __str__(self):
        return (
            f"{self.patient.upper()}\n"
            f"All Arcs: {self.all_arcs}\n"
            f"Current Arcs: {self.arcs}\n"
            f"All Control points: {self.full_control_point_count}\n"
            f"Current Control points: {self.control_point_count}\n"
            f"Energy: {self.energy}\n"
            f"Crop_size: {self.crop_size}\n"
            f"Dose Origin: {self.dose_origin}\n" 
            f"Dose Spacing: {self.dose_spacing}\n" 
            f"Dose Shape: {self.dose_shape}\n" 
        )

    def __len__(self):
        return self.control_point_count

for name in RaydoseComposer.TIMED_METHODS:
    if hasattr(RaydoseComposer, name):
        setattr(
            RaydoseComposer,
            name,
            timed_method(getattr(RaydoseComposer, name))
        )
