import os, glob

import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk

def read_nifti(file_path, print_info=True):
    sitk_image = sitk.ReadImage(file_path)
    if print_info:
        print("Loaded image:", file_path.split('/')[-1])
        print("Patient ID:", file_path.split('/')[-1].split('_')[0])
        
        if '_gtvt' in file_path: 
            modality = 'Binary GTV mask'
            sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)
        elif '_ct' in file_path: modality = 'CT'
        elif '_pt' in file_path: modality = 'PT' 
        print("Modality:", modality)
        
        image_size = sitk_image.GetSize()
        pixel_spacing = sitk_image.GetSpacing()
        print("Image size:", image_size)
        print("Pixel spacing (mm):", pixel_spacing)
        print("Physical size (mm):", [image_size[i]*pixel_spacing[i] for i in range(3)])


        image_stats = sitk.StatisticsImageFilter()
        image_stats.Execute(sitk_image)

        print(f"\n----- Image Statistics ----- \n Max Intensity: {image_stats.GetMaximum()} \
                \n Min Intensity: {image_stats.GetMinimum()} \n Mean: {image_stats.GetMean()} \
                \n Variance: {image_stats.GetVariance()} \n")
        
        print("Components per pixel:", sitk_image.GetNumberOfComponentsPerPixel())
        
        print("\n")
        
    return sitk_image

def mask_image_multiply(mask, image):
    components_per_pixel = image.GetNumberOfComponentsPerPixel()
    if  components_per_pixel == 1:
        return mask*image
    else:
        return sitk.Compose([mask*sitk.VectorIndexSelectionCast(image,channel) for channel in range(components_per_pixel)])

def alpha_blend(image1, image2, alpha = 0.5, mask1=None,  mask2=None):
    '''
    Alaph blend two images, pixels can be scalars or vectors.
    The alpha blending factor can be either a scalar or an image whose
    pixel type is sitkFloat32 and values are in [0,1].
    The region that is alpha blended is controled by the given masks.
    '''
    
    if not mask1:
        mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
        mask1.CopyInformation(image1)
    else:
        mask1 = sitk.Cast(mask1, sitk.sitkFloat32)
    if not mask2:
        mask2 = sitk.Image(image2.GetSize(),sitk.sitkFloat32) + 1
        mask2.CopyInformation(image2)
    else:        
        mask2 = sitk.Cast(mask2, sitk.sitkFloat32)
    # if we received a scalar, convert it to an image
    if type(alpha) != sitk.SimpleITK.Image:
        alpha = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + alpha
        alpha.CopyInformation(image1)
    components_per_pixel = image1.GetNumberOfComponentsPerPixel()
    if components_per_pixel>1:
        img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
        img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
    else:
        img1 = sitk.Cast(image1, sitk.sitkFloat32)
        img2 = sitk.Cast(image2, sitk.sitkFloat32)
        
    intersection_mask = mask1*mask2
    
    intersection_image = mask_image_multiply(alpha*intersection_mask, img1) + \
                         mask_image_multiply((1-alpha)*intersection_mask, img2)
    return intersection_image + mask_image_multiply(mask2-intersection_mask, img2) + \
           mask_image_multiply(mask1-intersection_mask, img1)

def display_slices(sitk_image, 
                   sagittal_slice_idxs=[], coronal_slice_idxs=[], axial_slice_idxs=[], 
                   window_level = None, window_width = None,
                   title=None, dpi=80):

    spacing = sitk_image.GetSpacing()
    

    if window_level != None and window_width != None:
        # Apply window and change scan image scale to 0-255
        print("windowing")
        window_min = window_level - window_width//2
        window_max = window_level + window_width//2
        sitk_image = sitk.Cast(sitk.IntensityWindowing(sitk_image, windowMinimum=window_min, windowMaximum=window_max, 
                                                       outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)
           
    ndarray = sitk.GetArrayFromImage(sitk_image)
    
    
    # Figure settings
    fig, [ax1,ax2,ax3] = plt.subplots(3)
    #fig.set_size_inches(0.5*18.5, 0.8*10.5)
    figsize = (2000/dpi, 1000/dpi)
    fig.set_size_inches(*figsize)
    fig.set_dpi(dpi)
    #fig.subplots_adjust(hspace=0.05, top=0.95, bottom=0.05, left=0.25, right=0.75)
    
    
    # Extract axial slices --
    axial_slices = []
    for idx in axial_slice_idxs:
        if ndarray.ndim == 3 : image2d = ndarray[idx, :, :]
        if ndarray.ndim == 4 : image2d = ndarray[idx, :, :, :]
        axial_slices.append(image2d)
    
    axial_slices = np.hstack(axial_slices)  

    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image height
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image width
    extent = (0, len(axial_slice_idxs)*n_cols*spacing[0], n_rows*spacing[1], 0)
    ax1.imshow(axial_slices, extent=extent, interpolation=None, cmap='gray')
    ax1.set_title(f"Axial slices: {axial_slice_idxs}")
    ax1.axis('off')
    
    
    # Extract coronal slices --
    coronal_slices = []
    for idx in coronal_slice_idxs:
        if ndarray.ndim == 3 : image2d = ndarray[:, idx, :]
        if ndarray.ndim == 4 : image2d = ndarray[:, idx, :, :]
        image2d = np.rot90(image2d, 2)
        coronal_slices.append(image2d)
        
    coronal_slices = np.hstack(coronal_slices)
    
    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image depth
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image width
    extent = (0, len(coronal_slice_idxs)*n_cols*spacing[0], n_rows*spacing[2], 0)
    ax2.imshow(coronal_slices, extent=extent, interpolation=None, cmap='gray')
    ax2.set_title(f"Coronal slices: {coronal_slice_idxs}")
    ax2.axis('off')

    
    # Extract sagittal slices --
    sagittal_slices = []
    for idx in sagittal_slice_idxs:
        if ndarray.ndim == 3 : image2d = ndarray[:, :, idx]
        if ndarray.ndim == 4 : image2d = ndarray[:, :, idx, :]
        image2d = np.rot90(image2d, k=2)
        image2d = np.flip(image2d, axis=1)
        sagittal_slices.append(image2d)
        
    sagittal_slices = np.hstack(sagittal_slices)
        
    n_rows = image2d.shape[0] # #rows of the 2d array - corresponds to sitk image depth
    n_cols = image2d.shape[1] # #columns of the 2d array - corresponds to sitk image height
    extent = (0, len(sagittal_slice_idxs)*n_cols*spacing[1], n_rows*spacing[2], 0)
    ax3.imshow(sagittal_slices, extent=extent, interpolation=None, cmap='gray')
    ax3.set_title(f"Sagittal slices: {sagittal_slice_idxs}")
    ax3.axis('off')
    
    if title:
        fig.suptitle(title, fontsize='x-large')
    plt.show()

    
    
    
def display_overlay_slices(sitk_image1, sitk_image2, sitk_image_seg,
                           sagittal_slice_idxs=[], coronal_slice_idxs=[], axial_slice_idxs=[], 
                           window1_level = None, window1_width = None, 
                           window2_level = None, window2_width = None, 
                           opacity=0.5, alpha=0.3,
                           title=None, dpi=80):

    # Apply window and change scan image scale to 0-255
    if window1_level != None and window1_width != None:

        window_min = window1_level - window1_width//2
        window_max = window1_level + window1_width//2
        sitk_image1 = sitk.Cast(sitk.IntensityWindowing(sitk_image1, windowMinimum=window_min, windowMaximum=window_max, 
                                                           outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

   # Apply window and change scan image scale to 0-255
    if sitk_image2 != None:

        if window2_level != None and window2_width != None:

            window_min = window2_level - window2_width//2
            window_max = window2_level + window2_width//2
            sitk_image2 = sitk.Cast(sitk.IntensityWindowing(sitk_image2, windowMinimum=window_min, windowMaximum=window_max, 
                                                               outputMinimum=0.0, outputMaximum=255.0), sitk.sitkUInt8)

        sitk_image = alpha_blend(sitk_image1, sitk_image2, alpha=alpha)  
        sitk_image = sitk.Cast(sitk_image, sitk.sitkUInt8)    
    else:
        sitk_image = sitk_image1
        
    # Overlay seg mask over the scan
    overlay_image = sitk.LabelOverlay(sitk_image, sitk_image_seg, opacity)

    # Display
    display_slices(overlay_image,
                           sagittal_slice_idxs, coronal_slice_idxs, axial_slice_idxs, 
                           window_level=None, window_width=None,
                           dpi=dpi, title=title)
        