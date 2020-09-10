import os
import numpy as np
import pandas as pd

import SimpleITK as sitk
from medpy.metric.binary import dc


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
	patient_ids = sorted(os.listdir(pred_2d_dir))

	dsc_dict = {}
	avg_dsc_2d = 0
	avg_dsc_3d = 0
	avg_dsc_ensemble_average = 0
	avg_dsc_ensemble_union = 0
	avg_dsc_ensemble_intersection = 0

	for p_id in patient_ids:
		gt_mask_path = pred_2d_dir + "/" + p_id + "/" + p_id + "_ct_gtvt.nii.gz"
		prob_2d_path = pred_2d_dir + "/" + p_id + "/predicted_ct_gtvt.nii.gz"
		prob_3d_path = pred_3d_dir + "/" + p_id + "/predicted_ct_gtvt.nii.gz"
		
		gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path))
		prob_map_2d = sitk.GetArrayFromImage(sitk.ReadImage(prob_2d_path))
		prob_map_3d = sitk.GetArrayFromImage(sitk.ReadImage(prob_3d_path))

		# Get binary masks using different ensembling strategies
		ensemble_average_mask = ensemble_average(prob_map_2d, prob_map_3d)
		ensemble_union_mask = ensemble_union(prob_map_2d, prob_map_3d)
		ensemble_intersection_mask = ensemble_intersection(prob_map_2d, prob_map_3d)

		# Compute DSC
		dsc_2d = dc(prob_map_2d>0.5, gt_mask)
		dsc_3d = dc(prob_map_3d>0.5, gt_mask)
		dsc_ensemble_average = dc(ensemble_average_mask, gt_mask)
		dsc_ensemble_union = dc(ensemble_union_mask, gt_mask)
		dsc_ensemble_intersection = dc(ensemble_intersection_mask, gt_mask)

		# Populate the patient-wise DSC dictionary
		dsc_dict[p_id] = {"2D": dsc_2d, "3D": dsc_3d, 
						  "Ensemble-Average": dsc_ensemble_average,
						  "Ensemble-Union": dsc_ensemble_union,
						  "Ensemble-Intersection": dsc_ensemble_intersection}

		# Accumulate the DSC values for calculating mean 
		avg_dsc_2d += dsc_2d
		avg_dsc_3d += dsc_3d
		avg_dsc_ensemble_average += dsc_ensemble_average
		avg_dsc_ensemble_union += dsc_ensemble_union
		avg_dsc_ensemble_intersection += dsc_ensemble_intersection

	# Calculate average DSC scores
	avg_dsc_2d /= len(patient_ids)
	avg_dsc_3d /= len(patient_ids)
	avg_dsc_ensemble_average /= len(patient_ids)
	avg_dsc_ensemble_union /= len(patient_ids)
	avg_dsc_ensemble_intersection /= len(patient_ids)

	dsc_dict["Average DSC"] = {"2D": avg_dsc_2d, "3D": avg_dsc_3d, 
								"Ensemble-Average": avg_dsc_ensemble_average,
								"Ensemble-Union": avg_dsc_ensemble_union,
								"Ensemble-Intersection": avg_dsc_ensemble_intersection}

	dsc_df = pd.DataFrame.from_dict(dsc_dict, orient="index")
	dsc_df.to_csv(output_path)



if __name__ == "__main__":
	# Default params
	pred_2d_dir = "../data/Ensemble/2D_Network"
	pred_3d_dir = "../data/Ensemble/3D_Network"
	output_path = "./validation_dsc.csv"

	main(pred_2d_dir, pred_3d_dir, output_path)