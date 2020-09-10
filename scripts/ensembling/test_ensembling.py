import os
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from pathlib import Path

import SimpleITK as sitk
from medpy.metric.binary import dc

import coloredlogs, logging
coloredlogs.install(level='INFO')
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def ensemble_average(prob_map_1, prob_map_2):
	voxelwise_avg = (prob_map_1 + prob_map_2) / 2
	return voxelwise_avg > 0.5

def ensemble_union(prob_map_1, prob_map_2):
	mask_1 = prob_map_1 > 0.5
	mask_2 = prob_map_2 > 0.5
	return np.maximum(mask_1, mask_2)

def ensemble_intersection(prob_map_1, prob_map_2):
	mask_1 = prob_map_1 > 0.5
	mask_2 = prob_map_2 > 0.5
	return mask_1 * mask_2


def main(pred_2d_dir, pred_3d_dir, output_path):
	patient_ids = sorted(os.listdir(str(pred_2d_dir)))

	for p_id in patient_ids:
		logger.info(f"Processing Patient: {p_id}")
		prob_2d_path = pred_2d_dir / p_id / "predicted_ct_gtvt.nii.gz"
		prob_3d_path = pred_3d_dir / p_id / "predicted_ct_gtvt.nii.gz"
		
		sitk_image_2d = sitk.ReadImage(str(prob_2d_path))
		sitk_image_3d = sitk.ReadImage(str(prob_3d_path))

		prob_map_2d = sitk.GetArrayFromImage(sitk_image_2d)
		prob_map_3d = sitk.GetArrayFromImage(sitk_image_3d)

		# Get binary masks using different ensembling strategies
		ensemble_results = {
			"average": ensemble_average(prob_map_2d, prob_map_3d),
			"union": ensemble_union(prob_map_2d, prob_map_3d),
			"intersection": ensemble_intersection(prob_map_2d, prob_map_3d)
		}

		for key, result in ensemble_results.items():
			patient_folder = output_path / key / p_id
			patient_folder.mkdir(parents=True, exist_ok=True)
			
			ensembled_mask = sitk.GetImageFromArray(result.astype(np.uint8))
			ensembled_mask.CopyInformation(sitk_image_2d)


			sitk.WriteImage(ensembled_mask, str(patient_folder / "predicted_ct_gtvt.nii.gz"))



if __name__ == "__main__":
	# Default params
	conf = OmegaConf.from_cli()

	pred_2d_dir = Path(conf.predictions_2d)
	pred_3d_dir = Path(conf.predictions_3d)
	output_path = Path(conf.output_path)

	main(pred_2d_dir, pred_3d_dir, output_path)