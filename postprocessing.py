import numpy as np
from scipy import ndimage


def get_cc_array(binary_array, structure_array=None):
    """
    Returns a label map with a unique integer label for each connected geometrical object in the given binary array.
    Integer labels of components start from 1. Background is 0.
    """

    if not structure_array:  # If not given, set 26-connected structure as default
        structure_array = np.array([
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
        # structure_array = ndimage.generate_binary_structure(rank=3, connectivity=3)

    cc_labeled_array, num_cc = ndimage.label(binary_array, structure=structure_array)

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
            - enhanced_gtv_mask_np: Numpy array of shape (Depth, Height, Width). Post-processed binary GTV mask
        """

        # Get label map of connected components
        pred_mask_cc_np = get_cc_array_scipy(pred_mask_np)


        # If enabled, assume brain-region false-positive as the largest geometric object and remove it
        if remove_brain_fp:
            cc_voxel_count = [np.sum(pred_mask_cc_np==label) for label in range(pred_mask_cc_np.max())]
            brain_fp_label = np.argmax(cc_voxel_count[1:]) + 1 # Get the label for the brain-region false positive object
            pred_mask_cc_np[pred_mask_cc_np == brain_fp_label] = 0 # Turn voxels with that label to zero


        # If not given, define a 5x5 structure matrix with a some-what spherical shape
        if not structure_matrix:
            structure_matrix = np.array([
                                         [[0,1,1,1,0],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [0,1,1,1,0]],

                                         [[0,1,1,1,0],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [0,1,1,1,0]],

                                         [[0,1,1,1,0],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [1,1,1,1,1],
                                          [0,1,1,1,0]]
                                        ])

        # Perform grey closing morphological operation to connect possibly multiple segmentations in the GTV region
        pred_mask_cc_closed_np = ndimage.morphology.grey_closing(pred_mask_cc_np, structure=structure_matrix)

        # Obtain the GTV as the largest geometric object
        cc_voxel_count = [np.sum(pred_mask_cc_closed_np==label) for label in range(pred_mask_cc_closed_np.max())]
        gtv_label = np.argmax(cc_voxel_count[1:]) + 1
        gtv_mask_np = pred_mask_cc_closed_np == gtv_label

        # Apply morphological binary closing on the GTV to make it more sperical
        enhanced_gtv_mask_np = ndimage.binary_closing(gtv_mask_np, structure=structure_matrix)

        return enhanced_gtv_mask_np