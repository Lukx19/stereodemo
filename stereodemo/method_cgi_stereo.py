from pathlib import Path
import shutil
import time
from dataclasses import dataclass
from typing import Optional
import urllib.request
import tempfile
import sys

import onnxruntime as rt

import cv2
import numpy as np
import torch

from .methods import Config, EnumParameter, StereoMethod, InputPair, StereoOutput
from . import utils

resource_url = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/358_CGI-Stereo/resources.tar.gz"
resource_filename = "resources.tar.gz"


class CGIStereo(StereoMethod):
    def __init__(self, config: Config):
        super().__init__("CGI Stereo (2022)",
                         "CGI Stereo",
                         {},
                         config)
        self.reset_defaults()

        self._loaded_session:Optional[rt.InferenceSession] = None
        self._loaded_model_path:Optional[Path] = None
        self._model_inputs = None
        self._model_outputs = None
        self._enable_profiling = False

    def reset_defaults(self):
        self.parameters.update ({
            "Shape": EnumParameter("Processed image size", 1, ["480x384", "640x480", "1280x736"]),
            "Training Set": EnumParameter("Dataset used during training", 0, ["sceneflow", "kitti"]),
        })

    def compute_disparity(self, input: InputPair) -> StereoOutput:
        cols, rows = self.parameters["Shape"].value.split('x')
        cols, rows = int(cols), int(rows)
        training_set = self.parameters["Training Set"].value

        model_path = self.config.models_path / f'cgistereo_{training_set}_{cols}x{rows}.onnx'
        self._load_model(model_path, training_set, rows, cols)
        assert self._loaded_session
        assert self._model_inputs
        assert self._model_outputs

        model_rows, model_cols = self._model_inputs[0].shape[2:] # B,C,H,W
        self.target_size = (model_cols, model_rows)

        combined_tensor = self._preprocess_input(input.left_image, input.right_image)
        left = combined_tensor[:,0:3, :, :]
        right = combined_tensor[:,3:6, :, :]

        start = time.time()
        outputs = self._loaded_session.run(['output'], { 'left': left, "right":right })
        elapsed_time = time.time() - start

        if self._enable_profiling:
            self._loaded_session.end_profiling()


        disparity_map = self._process_output(outputs)
        if disparity_map.shape[:2] != input.left_image.shape[:2]:
            disparity_map = cv2.resize(
                disparity_map,
                (input.left_image.shape[1], input.left_image.shape[0]),
                cv2.INTER_NEAREST)

            model_output_cols = disparity_map.shape[1]
            x_scale = input.left_image.shape[1] / float(model_output_cols)
            disparity_map *= np.float32(x_scale)

        return StereoOutput(disparity_map, input.left_image, elapsed_time)

    def _preprocess_input (self, left: np.ndarray, right: np.ndarray):
        left = cv2.resize(left, self.target_size, cv2.INTER_AREA)
        right = cv2.resize(right, self.target_size, cv2.INTER_AREA)

        # -> H,W,C=2 or 6 , normalized to [0,1]
        combined_img = np.concatenate((left, right), axis=-1) / 255.0
        # -> C,H,W
        combined_img = combined_img.transpose(2, 0, 1)
        # -> B=1,C,H,W
        combined_img = np.expand_dims(combined_img, 0).astype(np.float32)
        return combined_img

    def _process_output(self, outputs):
        disparity_map = outputs[0][0]
        return disparity_map

    def _load_model(self, model_path: Path, training_set, rows, cols):
        if (self._loaded_model_path == model_path):
            return

        if not model_path.exists():
            raw_models_folder =  model_path.parent / "cgistereo_raw"
            resources_filename = raw_models_folder / resource_filename
            if not resources_filename.exists():
                raw_models_folder.mkdir(exist_ok=True, parents=True)
                utils.donwload_file(resource_url, resource_filename, raw_models_folder)

            cgi_model_name = f"cgi_stereo_{training_set}_{rows}x{cols}"
            path_inside_tar = cgi_model_name + "/" + cgi_model_name+".onnx"
            extracted_filepath = raw_models_folder / Path(path_inside_tar)
            if not extracted_filepath.exists():
                utils.extract_file(
                    raw_models_folder / Path(resource_filename),
                    path_inside_tar,
                    raw_models_folder)
            shutil.copy(extracted_filepath, model_path)

        assert model_path.exists() and model_path.is_file()

        self._loaded_model_path = model_path
        sess_options = rt.SessionOptions()
        sess_options.enable_profiling = self._enable_profiling
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

        providers = []
        # providers.append(('TensorrtExecutionProvider',{'trt_fp16_enable':'1'}))
        providers.append(("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1', "cudnn_conv_algo_search": "HEURISTIC"}))
        providers.append(('CPUExecutionProvider', {}))

        self._loaded_session = rt.InferenceSession(str(model_path),
                                                   providers=providers, sess_options=sess_options)

        self._model_inputs = self._loaded_session.get_inputs()
        self._model_outputs = self._loaded_session.get_outputs()

        rows, cols = self._model_inputs[0].shape[2:] # B,C,H,W
        left = np.zeros((1, 3, rows, cols),np.float32)
        right = np.zeros((1, 3, rows, cols),np.float32)
        outputs = self._loaded_session.run(['output'], { 'left': left, "right":right })

