import os
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

import SimpleITK as sitk
from medpy.metric.binary import dc




def ensemble_weighted_average(prob_map_1, prob_map_2, weight_1=0.5):
    voxelwise_avg = weight_1*prob_map_1 + (1-weight_1)*prob_map_2
    return voxelwise_avg >= 0.5

def get_cc_array(binary_array, cc_structure=None):
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """

    if not cc_structure:  # If not given, set 26-connected structure as default
        cc_structure = np.array([
                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]],

                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]],

                                    [[1,1,1],
                                     [1,1,1],
                                     [1,1,1]]
                                   ])
        # or alternatively, use the following function -
        # cc_structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    cc_labeled_array, num_cc = ndimage.label(binary_array, structure=cc_structure)

    print("Number of connected components found:", num_cc)
    return cc_labeled_array


def extract_and_enhance_gtv_mask(pred_mask_np, structure_matrix=None, remove_brain_fp=False):
    """
    Post-processing function: Connected components + morphological operations
    Params:
        - pred_mask_np: Numpy array of shape (Depth, Height, Width). Binary GTV mask produced as the thresholded output of the neural-network
        - structure_matrix: Numpy array. Structuring element for morphological closing
        - remove_brain_fp: Bool.
    Returns:
        - gtv_mask_np: Numpy array of shape (Depth, Height, Width). Post-processed binary GTV mask
    """    # If not given, define a 5x5 structure matrix with an approximated spherical shape
    if not structure_matrix:
        structure_matrix = np.array([  [[0,0,1,0,0],
                                        [0,1,1,1,0],
                                        [1,1,1,1,1],
                                        [0,1,1,1,0],
                                        [0,0,1,0,0]],

                                       [[0,0,1,0,0],
                                        [0,1,1,1,0],
                                        [1,1,1,1,1],
                                        [0,1,1,1,0],
                                        [0,0,1,0,0]],

                                       [[0,0,1,0,0],
                                        [0,1,1,1,0],
                                        [1,1,1,1,1],
                                        [0,1,1,1,0],
                                        [0,0,1,0,0]]
                                    ])
    # Binary dilation to connect nearby components before connected components analysis
    pred_mask_np = ndimage.binary_dilation(pred_mask_np.astype(np.bool))

    # Get label map of connected components
    cc_array = get_cc_array(pred_mask_np)

    # Obtain the GTV as the largest geometric object
    voxel_count = [np.sum(cc_array==label) for label in range(1, cc_array.max() + 1)]

    if len(voxel_count) == 0:
        gtv_mask_np = pred_mask_np
    else:
        gtv_label = np.argmax(voxel_count) + 1
        gtv_mask_np = cc_array == gtv_label

    # Close obtained largest component to clean irregularities.
    gtv_mask_np = ndimage.binary_closing(gtv_mask_np, structure=structure_matrix)

    # Close any holes in the contours in n-d
    gtv_mask_np = ndimage.morphology.binary_fill_holes(gtv_mask_np, structure=structure_matrix)

    return gtv_mask_np



def main(pred_2d_dir, pred_3d_dir, output_path):
    patient_ids = sorted(os.listdir(pred_2d_dir))

    avg_dsc_per_weight = []
    avg_dsc_per_weight_postproc = []
    weight_list = np.arange(0, 1.1, 0.1)

    for weight in weight_list:

        avg_dsc = 0
        avg_dsc_postproc = 0

        for p_id in patient_ids:
            gt_mask_path = pred_2d_dir + "/" + p_id + "/" + p_id + "_ct_gtvt.nii.gz"
            prob_2d_path = pred_2d_dir + "/" + p_id + "/predicted_ct_gtvt.nii.gz"
            prob_3d_path = pred_3d_dir + "/" + p_id + "/predicted_ct_gtvt.nii.gz"

            gt_mask = sitk.GetArrayFromImage(sitk.ReadImage(gt_mask_path))
            prob_map_2d = sitk.GetArrayFromImage(sitk.ReadImage(prob_2d_path))
            pred_mask_2d = prob_map_2d >= 0.5
            prob_map_3d = sitk.GetArrayFromImage(sitk.ReadImage(prob_3d_path))
            pred_mask_3d = prob_map_3d >= 0.5

            # Get binary masks using different ensembling strategies
            ensemble_weighted_average_mask = ensemble_weighted_average(prob_map_2d, prob_map_3d, weight)

            # Prost-process and store in a separate variable
            ensemble_weighted_average_postproc_mask = extract_and_enhance_gtv_mask(ensemble_weighted_average_mask)

            # Compute dice
            dsc_ensemble_weighted_average = dc(ensemble_weighted_average_mask, gt_mask)
            dsc_ensemble_weighted_average_postproc = dc(ensemble_weighted_average_postproc_mask, gt_mask)
            avg_dsc += dsc_ensemble_weighted_average
            avg_dsc_postproc += dsc_ensemble_weighted_average_postproc

        avg_dsc /= len(patient_ids)
        avg_dsc_per_weight.append(avg_dsc)

        avg_dsc_postproc /= len(patient_ids)
        avg_dsc_per_weight_postproc.append(avg_dsc_postproc)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(weight_list, avg_dsc_per_weight, 'go-', label="Without post-processing")
    ax.plot(weight_list, avg_dsc_per_weight_postproc, 'mo-', label="With post-processing")
    ax.set_xlabel("Weight on 3D-to-2D U-Net")
    ax.set_ylabel("Average validation Dice score")

    # Show the peak for non-post-proc model
    ax.plot([-0.1, weight_list[np.argmax(avg_dsc_per_weight)]],
    	     [max(avg_dsc_per_weight), max(avg_dsc_per_weight)],
    	     'g--', alpha=0.45)
    ax.plot([weight_list[np.argmax(avg_dsc_per_weight)], weight_list[np.argmax(avg_dsc_per_weight)]],
    	     [0.620, max(avg_dsc_per_weight)],
    	     'g--', alpha=0.45)

    # Show the peak for post-proc model
    ax.plot([-0.1, weight_list[np.argmax(avg_dsc_per_weight_postproc)]],
    	     [max(avg_dsc_per_weight_postproc), max(avg_dsc_per_weight_postproc)],
    	     'm--', alpha=0.45)
    ax.plot([weight_list[np.argmax(avg_dsc_per_weight_postproc)], weight_list[np.argmax(avg_dsc_per_weight_postproc)]],
    	     [0.620, max(avg_dsc_per_weight_postproc)],
    	     'm--', alpha=0.45)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.620, 0.665)

    ax.legend()
    ax.grid()
    plt.savefig("ensemble_weight_vs_dice_plot.png")
    plt.show()



if __name__ == "__main__":
    # Default params
    pred_2d_dir = "../data/Ensemble/2d_network_augmented"
    pred_3d_dir = "../data/Ensemble/3D_Network"
    output_path = "./ensemble_weighted_avg_weightVdsc.png"

    main(pred_2d_dir, pred_3d_dir, output_path)