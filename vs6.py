#!/usr/bin/env python
"""
Brain Template Registration for 5 Patients

This script performs the registration of individual brain imaging data 
with a brain template for 5 patients. It uses the following components:
1. Patch Generation
2. Model Training 
3. Single-fold Testing
4. Five-fold Weighted Average Ensembling

Copyright (c) 2025 Medical Imaging Research
Version: 1.0.0
"""

import os
import sys
import numpy as np
import nibabel as nib
import h5py
import glob 
import tensorflow as tf
import tensorflow_addons as tfa
from skimage import measure, transform
from skimage.morphology import reconstruction
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
import argparse
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as skimage_ssim
from collections import defaultdict
import json
from datetime import datetime
from skimage.transform import resize

# Set version information
__version__ = "1.0.0"

class BrainRegistration:
    """Class for brain template registration"""
    
    def __init__(self, gpu_id="0"):
        """Initialize brain registration model with GPU settings"""
        # Set GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        
        # Initialize model
        self.model = None
        
        # Set up logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Set up logging for the registration process"""
        self.log_messages = []
        
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.log_messages.append(log_entry)
        
    def save_log(self, filepath):
        """Save log messages to a file"""
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.log_messages))
        
    def apply_deformation_enhanced(self, args):
        """
        Enhanced deformation application with parameter-specific handling
        
        Args:
            args: Tuple containing moving_image and deformation_field
            
        Returns:
            Warped image
        """
        moving_image, deformation_field = args
        
        # Add epsilon to prevent division by zero
        epsilon = 1e-8
        
        # Scale deformation field and clip to reasonable range
        deformation_field = tf.clip_by_value(deformation_field * 20.0, -100.0, 100.0)
        
        moving_shape = K.shape(moving_image)
        
        # Parameter-specific scaling
        scales = [1.0, 0.8, 0.6, 0.6]  # CBF, CBV, Tmax, MTT
        
        warped_channels = []
        for i in range(4):
            scaled_def = deformation_field * scales[i]
            
            # Create sampling grid with epsilon to prevent edge cases
            height, width = moving_shape[1], moving_shape[2]
            x_linspace = tf.linspace(0.0, tf.cast(width, 'float32') - 1.0, width)
            y_linspace = tf.linspace(0.0, tf.cast(height, 'float32') - 1.0, height)
            x_coords, y_coords = tf.meshgrid(x_linspace, y_linspace)
            
            # Apply scaled deformation with clipping to prevent out-of-bounds values
            x_warped = tf.clip_by_value(x_coords + scaled_def[..., 0], 0.0, tf.cast(width, 'float32') - 1.0)
            y_warped = tf.clip_by_value(y_coords + scaled_def[..., 1], 0.0, tf.cast(height, 'float32') - 1.0)
            
            # Normalize coordinates safely
            x_warped = 2.0 * x_warped / (tf.cast(width, 'float32') - 1.0 + epsilon) - 1.0
            y_warped = 2.0 * y_warped / (tf.cast(height, 'float32') - 1.0 + epsilon) - 1.0
            
            # Stack coordinates and sample
            coords = tf.stack([x_warped, y_warped], axis=-1)
            
            # Use resampler for interpolation
            warped_channel = tfa.image.resampler(moving_image[..., i:i+1], coords)
            
            # Handle NaNs in output
            warped_channel = tf.where(tf.math.is_nan(warped_channel), 
                                    moving_image[..., i:i+1], 
                                    warped_channel)
            
            warped_channels.append(warped_channel)
        
        # Combine warped channels
        warped_image = K.concatenate(warped_channels, axis=-1)
        warped_image = tf.image.resize(warped_image, (moving_shape[1], moving_shape[2]), method='bilinear')
        
        # Final NaN check
        warped_image = tf.where(tf.math.is_nan(warped_image), moving_image, warped_image)
        
        return warped_image

    def channel_attention(self, feature, name=None):
        """
        Apply channel attention mechanism
        
        Args:
            feature: Input feature map
            name: Optional name prefix for layers
            
        Returns:
            Attention-weighted feature map
        """
        ratio = 4
        channel = K.int_shape(feature)[-1]
        
        # Create base name if none provided
        base_name = name if name else f'channel_attention_{K.get_uid("channel_attention")}'
        
        feature = tf.keras.layers.SpatialDropout2D(rate=0.2, 
                                data_format="channels_last", 
                                name=f'{base_name}_dropout')(feature)
        avg_pool = tf.keras.layers.GlobalAveragePooling2D(name=f'{base_name}_gap')(feature)
        
        avg_pool = tf.keras.layers.Reshape((1, 1, channel), 
                        name=f'{base_name}_reshape')(avg_pool)
        
        reduced_channels = max(channel // ratio, 1)
        
        # First branch
        avg_pool1 = tf.keras.layers.Dense(reduced_channels, 
                        activation='relu', 
                        use_bias=True, 
                        bias_initializer='zeros',
                        name=f'{base_name}_dense1_1')(avg_pool)
        avg_pool1 = tf.keras.layers.Dense(channel, 
                        use_bias=True, 
                        bias_initializer='zeros',
                        name=f'{base_name}_dense1_2')(avg_pool1)
        
        # Second branch
        avg_pool2 = tf.keras.layers.Dense(reduced_channels, 
                        activation='relu', 
                        use_bias=True, 
                        bias_initializer='zeros',
                        name=f'{base_name}_dense2_1')(avg_pool)
        avg_pool2 = tf.keras.layers.Dense(channel, 
                        use_bias=True, 
                        bias_initializer='zeros',
                        name=f'{base_name}_dense2_2')(avg_pool2)
        
        feature1 = tf.keras.layers.Add(name=f'{base_name}_add')([avg_pool1, avg_pool2])
        feature1 = tf.keras.layers.Activation('sigmoid', name=f'{base_name}_sigmoid')(feature1)
        
        return tf.keras.layers.multiply([feature, feature1], name=f'{base_name}_multiply')

    def create_convolution_block(self, input_layer, n_filters, kernel_size=3, strides=1, 
                               activation='relu', instance_normalization=True, weight_decay=0.0001):
        """
        Create a convolutional block with instance normalization
        
        Args:
            input_layer: Input tensor
            n_filters: Number of filters
            kernel_size: Size of convolutional kernel
            strides: Stride size
            activation: Activation function
            instance_normalization: Whether to use instance normalization
            weight_decay: Weight decay factor
            
        Returns:
            Convolutional block
        """
        conv = tf.keras.layers.Conv2D(n_filters, kernel_size, strides=strides, padding='same', 
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_layer)
        if instance_normalization:
            conv = tfa.layers.InstanceNormalization()(conv)
        if activation is None:
            return conv
        return tf.keras.layers.Activation(activation)(conv)

    def create_context_module(self, input_layer, n_level_filters, dropout_rate=0.3):
        """
        Create a context module with spatial dropout
        
        Args:
            input_layer: Input tensor
            n_level_filters: Number of filters
            dropout_rate: Dropout rate
            
        Returns:
            Context module
        """
        conv1 = self.create_convolution_block(input_layer, n_level_filters)
        dropout = tf.keras.layers.SpatialDropout2D(rate=dropout_rate)(conv1)
        conv2 = self.create_convolution_block(dropout, n_level_filters)
        return conv2

    def create_registration_model(self, fixed_input_shape=(64, 64, 1), moving_input_shape=(64, 64, 4), 
                                n_base_filters=16, depth=4, dropout_rate=0.3):
        """
        Create the registration model
        
        Args:
            fixed_input_shape: Shape of fixed input (template)
            moving_input_shape: Shape of moving input (patient data)
            n_base_filters: Base number of filters
            depth: Depth of U-Net
            dropout_rate: Dropout rate
            
        Returns:
            Registration model
        """
        fixed_input = Input(fixed_input_shape, name='fixed_input')
        moving_input = Input(moving_input_shape, name='moving_input')
        
        fixed_input_shape = Lambda(lambda x: x, name='fixed_input_shape')(fixed_input)
        moving_input_shape = Lambda(lambda x: x, name='moving_input_shape')(moving_input)
        
        # Encoder path
        current_layer = tf.keras.layers.Concatenate(name='initial_concat')([fixed_input, moving_input])
        
        level_outputs = []
        for level in range(depth):
            n_level_filters = (2**level) * n_base_filters
            
            # Convolution block
            conv = self.create_convolution_block(current_layer, n_level_filters)
            
            # Context module
            context = self.create_context_module(conv, n_level_filters, dropout_rate)
            
            # Feature fusion with unique names
            current_layer = tf.keras.layers.Add(name=f'encoder_add_level_{level}')([conv, context])
            current_layer = self.channel_attention(current_layer, name=f'encoder_attention_{level}')
            
            level_outputs.append(current_layer)
            
            if level < depth - 1:
                current_layer = self.create_convolution_block(current_layer, n_level_filters, strides=2)
        
        # Decoder path with size management
        for level in range(depth-2, -1, -1):
            n_level_filters = (2**level) * n_base_filters
            
            # Get skip connection
            skip_tensor = level_outputs[level]
            skip_shape = K.int_shape(skip_tensor)
            
            # Upsampling path
            up = tf.keras.layers.UpSampling2D(size=(2, 2), name=f'decoder_upsample_{level}')(current_layer)
            up = Conv2D(n_level_filters, 3, padding='same', name=f'decoder_conv_{level}')(up)
            
            # Ensure both tensors have same dimensions
            current_shape = K.int_shape(up)
            if current_shape[1:3] != skip_shape[1:3]:
                up = Lambda(lambda x: tf.image.resize(x, 
                                                    (skip_shape[1], skip_shape[2]), 
                                                    method='bilinear'), 
                           name=f'decoder_resize_{level}')(up)
            
            # Concatenate features
            concat = tf.keras.layers.Concatenate(name=f'decoder_concat_{level}')([up, skip_tensor])
            
            # Process concatenated features
            current_layer = self.create_convolution_block(concat, n_level_filters)
            current_layer = self.create_context_module(current_layer, n_level_filters, dropout_rate)
            current_layer = self.channel_attention(current_layer, name=f'decoder_attention_{level}')

        # Parameter branches
        def create_parameter_branch(input_layer, name):
            x = Conv2D(32, 3, padding='same', name=f'branch_{name}_conv1')(input_layer)
            x = BatchNormalization(name=f'branch_{name}_bn1')(x)
            x = Activation('relu', name=f'branch_{name}_relu1')(x)
            x = Conv2D(16, 3, padding='same', name=f'branch_{name}_conv2')(x)
            x = BatchNormalization(name=f'branch_{name}_bn2')(x)
            x = Activation('relu', name=f'branch_{name}_relu2')(x)
            x = Conv2D(2, 3, activation='tanh', padding='same', name=f'branch_{name}_final')(x)
            return x

        # Create branches with unique names
        main_deformation = create_parameter_branch(current_layer, 'main')
        cbf_branch = create_parameter_branch(current_layer, 'cbf')
        cbv_branch = create_parameter_branch(current_layer, 'cbv')
        tmax_branch = create_parameter_branch(current_layer, 'tmax')
        mtt_branch = create_parameter_branch(current_layer, 'mtt')

        # Combine deformations
        weights = [1.0, 0.8, 0.6, 0.6]
        deformations = [main_deformation, cbf_branch, cbv_branch, tmax_branch, mtt_branch]
        combined = sum(d * w for d, w in zip(deformations, [1.0] + weights))

        deformation_field = Lambda(lambda x: tf.image.resize(x, (64, 64), method='bilinear'),
                                 name='final_deformation_resize')(combined)
        
        # Add a scaling layer to increase deformation magnitude
        scale_factor = 15.0  # Large scaling factor to ensure visible deformation
        deformation_field = Lambda(lambda x: x * scale_factor, name='deformation_scaling')(deformation_field)
        
        # Apply deformation
        warped_moving = Lambda(self.apply_deformation_enhanced, name='deform_apply')([moving_input, deformation_field])

        # Final processing
        target_shape = (64, 64)
        fixed_resized = Lambda(lambda x: tf.image.resize(x, target_shape, method='bilinear'),
                             name='fixed_final_resize')(fixed_input)
        warped_resized = Lambda(lambda x: tf.image.resize(x, target_shape, method='bilinear'),
                              name='warped_final_resize')(warped_moving)
        def_resized = Lambda(lambda x: tf.image.resize(x, target_shape, method='bilinear'),
                           name='def_final_resize')(deformation_field)

        # Shape enforcement
        fixed_resized = Lambda(lambda x: K.reshape(x, (-1, 64, 64, 1)), name='fixed_reshape')(fixed_resized)
        warped_resized = Lambda(lambda x: K.reshape(x, (-1, 64, 64, 4)), name='warped_reshape')(warped_resized)
        def_resized = Lambda(lambda x: K.reshape(x, (-1, 64, 64, 2)), name='def_reshape')(def_resized)

        # Output
        outputs = tf.keras.layers.Concatenate(name='final_concat')([fixed_resized, warped_resized, def_resized])
        outputs = Conv2D(7, 1, activation='sigmoid', name='final_conv')(outputs)
        outputs = Lambda(lambda x: K.reshape(x, (-1, 64, 64, 7)), name='final_reshape')(outputs)

        # Model
        model = Model(inputs=[fixed_input, moving_input], outputs=outputs)
        
        opt = Adam(learning_rate=1e-4, clipnorm=1.0, clipvalue=0.5)
        model.compile(optimizer=opt, 
                     loss=self.combined_loss, 
                     metrics=[self.mse_metric, self.ssim_metric, self.dice_coefficient, self.jacobian_determinant])
        
        self.model = model
        return model

    # Loss functions and metrics
    def safe_mean(self, x):
        """Calculate mean with NaN handling"""
        return tf.reduce_mean(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))

    def mse_metric(self, y_true, y_pred):
        """Calculate MSE metric between fixed and warped images"""
        fixed = y_pred[..., 0:1]
        mse_scores = []
        for i in range(4):
            warped = y_pred[..., i+1:i+2]
            squared_diff = tf.reduce_mean(tf.square(fixed - warped), axis=[1, 2])
            mse_scores.append(squared_diff)
        return tf.reduce_mean(tf.stack(mse_scores))

    def ssim_metric(self, y_true, y_pred):
        """Calculate SSIM metric between fixed and warped images"""
        fixed = y_pred[..., 0:1]
        ssim_scores = []
        for i in range(4):
            warped = y_pred[..., i+1:i+2]
            score = tf.image.ssim(fixed, warped, max_val=1.0)
            ssim_scores.append(score)
        return tf.reduce_mean(tf.stack(ssim_scores))

    def dice_coefficient(self, y_true, y_pred):
        """Calculate Dice coefficient between fixed and warped images"""
        fixed = y_pred[..., 0:1]
        dice_scores = []
        for i in range(4):
            warped = y_pred[..., i+1:i+2]
            intersection = tf.reduce_sum(fixed * warped, axis=[1, 2])
            sum_fixed = tf.reduce_sum(fixed, axis=[1, 2])
            sum_warped = tf.reduce_sum(warped, axis=[1, 2])
            dice = (2. * intersection + 1e-5) / (sum_fixed + sum_warped + 1e-5)
            dice_scores.append(dice)
        return tf.reduce_mean(tf.stack(dice_scores))

    def jacobian_determinant(self, y_true, y_pred):
        """Calculate Jacobian determinant of deformation field"""
        deformation = y_pred[..., -2:]  # Shape: (batch, H, W, 2)
        
        # Calculate spatial gradients
        dy_x, dx_x = tf.image.image_gradients(deformation[..., 0:1])  # x-component
        dy_y, dx_y = tf.image.image_gradients(deformation[..., 1:2])  # y-component
        
        # Compute Jacobian determinant
        det = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
        
        return tf.reduce_mean(det)

    def smoothness_loss(self, deformation):
        """Calculate deformation field smoothness loss"""
        dx = deformation[:, 1:, :, :] - deformation[:, :-1, :, :]
        dy = deformation[:, :, 1:, :] - deformation[:, :, :-1, :]
        
        dx = tf.pad(dx, [[0, 0], [0, 1], [0, 0], [0, 0]])
        dy = tf.pad(dy, [[0, 0], [0, 0], [0, 1], [0, 0]])
        
        return K.mean(K.square(dx) + K.square(dy))

    def combined_loss(self, y_true, y_pred):
        """Combined loss function for registration model"""
        # Get components with NaN checking
        fixed = y_pred[..., 0:1]
        warped = y_pred[..., 1:5]
        deformation = y_pred[..., -2:]
        
        # Calculate metrics with NaN protection
        mse_val = tf.where(tf.math.is_nan(self.mse_metric(y_true, y_pred)), 
                         tf.constant(1.0, dtype=tf.float32), 
                         self.mse_metric(y_true, y_pred))
        
        # Calculate deformation magnitude with NaN handling
        deform_magnitude = tf.sqrt(tf.reduce_sum(tf.square(deformation) + 1e-8, axis=-1))
        deform_mean = self.safe_mean(deform_magnitude)
        
        # Add penalty term that's NaN safe
        deform_penalty = tf.maximum(0.0, 1.0 - tf.where(tf.math.is_nan(deform_mean), 
                                                       tf.constant(0.0, dtype=tf.float32), 
                                                       deform_mean))
        
        # Calculate smoothness with NaN protection
        smooth_val = tf.where(tf.math.is_nan(self.smoothness_loss(deformation)), 
                            tf.constant(0.1, dtype=tf.float32), 
                            self.smoothness_loss(deformation))
        
        # Combined loss with NaN protection
        total_loss = mse_val + 0.5 * deform_penalty + 0.1 * smooth_val
        return tf.where(tf.math.is_nan(total_loss), tf.constant(1.0, dtype=tf.float32), total_loss)

    def safe_normalize(self, image):

     min_val = np.min(image)
     max_val = np.max(image)
    
     if max_val == min_val:
        return np.zeros_like(image)
    
     return (image - min_val) / (max_val - min_val)
     
    def resize_to_match(self, template_slice, target_shape):


    
     if template_slice.shape != target_shape:
        return resize(
            template_slice,
            target_shape,
            order=1,  # Linear interpolation
            mode='constant',
            anti_aliasing=True
        )
     return template_slice 

    def calculate_slice_metrics(self, orig_slice, reg_slice):

     if orig_slice.shape != reg_slice.shape:
        # Resize original to match registered slice size
        orig_slice = self.resize_to_match(orig_slice, reg_slice.shape)
    
    # Normalize slices
     orig_norm = self.safe_normalize(orig_slice)
     reg_norm = self.safe_normalize(reg_slice)
    
    # Calculate MSE
     mse = np.mean((orig_norm - reg_norm) ** 2)
    
    # Ensure slices have correct dimensions for SSIM
     if orig_norm.ndim < 2 or reg_norm.ndim < 2:
        return mse, 0.0
        
    # Calculate SSIM with dynamic window size
     win_size = min(7, orig_norm.shape[0] - 1, orig_norm.shape[1] - 1)
     win_size = max(win_size, 3)  # Ensure minimum window size of 3
    
     if win_size % 2 == 0:  # SSIM requires odd window size
        win_size += 1
        
     try:
        ssim_value = skimage_ssim(orig_norm, reg_norm, data_range=1.0, win_size=win_size)
     except Exception as e:
        print(f"SSIM calculation failed: {e}")
        ssim_value = 0.0
        
     return mse, ssim_value

    def create_checkerboard(self, img1, img2, squares=8):

     if img1.shape != img2.shape:
        img1 = self.resize_to_match(img1, img2.shape)
    
    # Create checkerboard pattern
     h, w = img1.shape
     sq_h, sq_w = h // squares, w // squares
    
    # Initialize result with img1
     result = img1.copy()
    
    # Create checkerboard mask
     for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 1:  # Alternate squares
                # Define square region
                r_start, r_end = i * sq_h, min((i + 1) * sq_h, h)
                c_start, c_end = j * sq_w, min((j + 1) * sq_w, w)
                # Replace with img2 in this square
                result[r_start:r_end, c_start:c_end] = img2[r_start:r_end, c_start:c_end]
    
     return result

    def save_comparison_image(self, template_slice, registered_slice, original_slice, slice_metrics, save_path, modality_name):

     if template_slice.shape != registered_slice.shape:
        template_slice = self.resize_to_match(template_slice, registered_slice.shape)
    
    # Normalize slices for visualization
     template_norm = self.safe_normalize(template_slice)
     original_norm = self.safe_normalize(original_slice)
     registered_norm = self.safe_normalize(registered_slice)
    
    # Create a figure with 2x3 subplots
     fig = plt.figure(figsize=(18, 10))
    
    # Add a title to the whole figure
     fig.suptitle(f"Template-Based Registration Results - {modality_name}", fontsize=16)
    
    # Template image (fixed)
     ax1 = plt.subplot(231)
     im1 = ax1.imshow(template_norm, cmap='gray', vmin=0, vmax=1)
     ax1.set_title('IIT Template (Fixed)', fontsize=12)
     ax1.set_axis_off()
    
    # Original modality (moving)
     ax2 = plt.subplot(232)
     im2 = ax2.imshow(original_norm, cmap='gray', vmin=0, vmax=1)
     ax2.set_title(f'Original {modality_name}', fontsize=12)
     ax2.set_axis_off()
    
    # Registered modality
     ax3 = plt.subplot(233)
     im3 = ax3.imshow(registered_norm, cmap='gray', vmin=0, vmax=1)
     ax3.set_title('Template-Registered Result', fontsize=12)
     ax3.set_axis_off()
    
    # Difference between original and registered
     ax4 = plt.subplot(234)
     diff = np.abs(original_norm - registered_norm)
     im4 = ax4.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
     ax4.set_title('Original vs Registered Difference', fontsize=12)
     ax4.set_axis_off()
     plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Red-Green Overlay (registered=red, template=green)
     ax5 = plt.subplot(235)
     overlay_rg = np.zeros((registered_norm.shape[0], registered_norm.shape[1], 3))
     overlay_rg[..., 0] = registered_norm  # Red channel = registered
     overlay_rg[..., 1] = template_norm    # Green channel = template
     im5 = ax5.imshow(overlay_rg)
     ax5.set_title('Overlay: Registered(R) + Template(G)', fontsize=12)
     ax5.set_axis_off()
    
    # Checkerboard pattern
     ax6 = plt.subplot(236)
     checkerboard = self.create_checkerboard(template_norm, registered_norm, 8)
     im6 = ax6.imshow(checkerboard, cmap='gray')
     ax6.set_title('Checkerboard View', fontsize=12)
     ax6.set_axis_off()
    
    # Add metrics as text with clear background
     metrics_text = (f'SSIM: {slice_metrics["ssim"]:.4f}   MSE: {slice_metrics["mse"]:.6f}\n'
                   f'Higher SSIM and lower MSE indicate better registration')
     plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=5))
    
    # Adjust layout and save with high quality
     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
     plt.savefig(save_path, dpi=300, bbox_inches='tight')
     plt.close()

    def create_difference_map(self, original_slice, registered_slice, save_path):

     original_norm = self.safe_normalize(original_slice)
     registered_norm = self.safe_normalize(registered_slice)
    
    # Calculate absolute difference
     diff = np.abs(original_norm - registered_norm)
    
    # Create figure
     plt.figure(figsize=(10, 8))
    
    # Display difference map with colorbar
     plt.imshow(diff, cmap='hot', vmin=0, vmax=0.5)
     plt.colorbar(label='Absolute Difference')
     plt.title('Difference Map: Original vs Registered')
     plt.axis('off')
    
    # Save with high quality
     plt.tight_layout()
     plt.savefig(save_path, dpi=300, bbox_inches='tight')
     plt.close()

    def create_summary_visualization(self, subject_id, modality_name, metrics, slice_indices, save_path):

     plt.figure(figsize=(12, 10))
    
    # Plot MSE
     plt.subplot(211)
     plt.plot(slice_indices, metrics['mse'], 'ro-', linewidth=2)
     plt.title(f'Mean Squared Error - Subject {subject_id} - {modality_name}', fontsize=14)
     plt.xlabel('Slice Index', fontsize=12)
     plt.ylabel('MSE (lower is better)', fontsize=12)
     plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot SSIM
     plt.subplot(212)
     plt.plot(slice_indices, metrics['ssim'], 'bo-', linewidth=2)
     plt.title(f'Structural Similarity Index - Subject {subject_id} - {modality_name}', fontsize=14)
     plt.xlabel('Slice Index', fontsize=12)
     plt.ylabel('SSIM (higher is better)', fontsize=12)
     plt.grid(True, linestyle='--', alpha=0.7)
    
    # Overall metrics
     avg_mse = np.mean(metrics['mse'])
     avg_ssim = np.mean(metrics['ssim'])
    
    # Add text box with average metrics
     plt.figtext(0.5, 0.01, 
               f"Average MSE: {avg_mse:.6f}   Average SSIM: {avg_ssim:.4f}",
               ha='center', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=5))
    
     plt.tight_layout(rect=[0, 0.03, 1, 0.97])
     plt.savefig(save_path, dpi=300, bbox_inches='tight')
     plt.close()

    def create_registration_montage(self, template_slices, original_slices, registered_slices, modality_name, save_path, max_slices=16):

     n_slices = min(len(template_slices), len(original_slices), len(registered_slices), max_slices)
     if n_slices == 0:
        print("No slices available for montage.")
        return
    
    # Calculate grid size for montage
     cols = min(4, n_slices)
     rows = (n_slices + cols - 1) // cols  # Ceiling division
    
    # Create figure
     fig = plt.figure(figsize=(cols * 5, rows * 4))
    
    # Main title
     fig.suptitle(f"{modality_name} Registration Montage", fontsize=16)
    
    # Select slices to display (evenly sampled)
     step = max(1, len(template_slices) // n_slices)
     slice_indices = list(range(0, len(template_slices), step))[:n_slices]
    
     for i, slice_idx in enumerate(slice_indices):
        if i >= n_slices:
            break
            
        template_slice = template_slices[slice_idx]
        original_slice = original_slices[slice_idx]
        registered_slice = registered_slices[slice_idx]
        
        # Resize template to match registered dimensions if needed
        if template_slice.shape != registered_slice.shape:
            template_slice = self.resize_to_match(template_slice, registered_slice.shape)
        
        # Normalize slices
        template_norm = self.safe_normalize(template_slice)
        original_norm = self.safe_normalize(original_slice)
        registered_norm = self.safe_normalize(registered_slice)
        
        # Create RGB overlay
        overlay = np.zeros((template_norm.shape[0], template_norm.shape[1], 3))
        overlay[..., 0] = registered_norm  # Red channel = registered
        overlay[..., 1] = template_norm    # Green channel = template
        
        # Add subplots
        plt.subplot(rows, cols, i + 1)
        plt.imshow(overlay)
        plt.title(f"Slice {slice_idx}")
        plt.axis('off')
    
     plt.tight_layout(rect=[0, 0, 1, 0.95])
     plt.savefig(save_path, dpi=300, bbox_inches='tight')
     plt.close()

    def create_alignment_summary(self, template_data, modality_data, registered_data, modality_name, save_path, max_slices=9):

     available_slices = min(len(template_data), len(modality_data), len(registered_data))
    
     if available_slices == 0:
        print(f"No common slices available for alignment summary for {modality_name}")
        return
    
    # Determine number of slices to display
     n_slices = min(available_slices, max_slices)
    
    # Sample slices evenly
     step = max(1, available_slices // n_slices)
     slice_indices = range(0, available_slices, step)[:n_slices]
    
    # Skip if too few slices
     if len(slice_indices) <= 1:
        print(f"Only {len(slice_indices)} slice(s) available for alignment summary for {modality_name}")
        if len(slice_indices) == 0:
            return
    
    # Create figure - adjust rows if needed
     rows = len(slice_indices)
     fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    
    # Handle case with single row
     if rows == 1:
        axes = [axes]
    
     fig.suptitle(f"Alignment Summary: {modality_name}", fontsize=16)
    
    # Column titles
     col_titles = ['Template', 'Original', 'Registered', 'Overlay']
     for i, title in enumerate(col_titles):
        fig.text(0.125 + i*0.21, 0.95, title, ha='center', fontsize=14, fontweight='bold')
    
    # Process each slice
     for row, slice_idx in enumerate(slice_indices):
        # Get slices
        template_slice = template_data[slice_idx]
        original_slice = modality_data[slice_idx]
        registered_slice = registered_data[slice_idx]
        
        if template_slice.shape != registered_slice.shape:
            template_slice = self.resize_to_match(template_slice, registered_slice.shape)
        
        # Normalize slices
        template_norm = self.safe_normalize(template_slice)
        original_norm = self.safe_normalize(original_slice)
        registered_norm = self.safe_normalize(registered_slice)
        
        # Create overlay
        overlay = np.zeros((template_norm.shape[0], template_norm.shape[1], 3))
        overlay[..., 0] = registered_norm  # Red channel = registered
        overlay[..., 1] = template_norm    # Green channel = template
        
        # Display images
        axes[row][0].imshow(template_norm, cmap='gray')
        axes[row][0].set_title(f"Slice {slice_idx}")
        axes[row][0].axis('off')
        
        axes[row][1].imshow(original_norm, cmap='gray')
        axes[row][1].axis('off')
        
        axes[row][2].imshow(registered_norm, cmap='gray')
        axes[row][2].axis('off')
        
        axes[row][3].imshow(overlay)
        axes[row][3].axis('off')
        
        # Calculate metrics for each slice
        mse = np.mean((template_norm - registered_norm) ** 2)
        ssim_val = skimage_ssim(template_norm, registered_norm, data_range=1.0)
        
        # Add metrics as text
        metrics_text = f"MSE: {mse:.4f}, SSIM: {ssim_val:.4f}"
        axes[row][3].text(10, 10, metrics_text, color='white', 
                         bbox=dict(facecolor='black', alpha=0.7),
                         fontsize=8)
    
     plt.tight_layout(rect=[0, 0, 1, 0.95])
     plt.savefig(save_path, dpi=300, bbox_inches='tight')
     plt.close()

    def make_compatible_slices(template_data, ct_num_slices):

     print(f"Adjusting template slices from {len(template_data)} to {ct_num_slices}")
    
    # If template has more slices, sample them evenly
     if len(template_data) > ct_num_slices:
        indices = np.linspace(0, len(template_data)-1, ct_num_slices, dtype=int)
        return [template_data[i] for i in indices]
    
    # If template has fewer slices, duplicate slices as needed
     elif len(template_data) < ct_num_slices:
        result = []
        for i in range(ct_num_slices):
            idx = min(i, len(template_data)-1)  # Use last slice if we run out
            result.append(template_data[idx])
        return result
    
    # If same number of slices, return as is
     return template_data

    def detailed_slice_diagnosis(data, modality_name='Unknown'):

     print(f"\n--- Detailed {modality_name} Slice Diagnosis ---")
     print(f"Data Shape: {data.shape}")
    
    # Slice-wise detailed analysis
     for slice_idx in range(data.shape[0]):
        slice_data = data[slice_idx]
        
        print(f"\nSlice {slice_idx} Analysis:")
        print("Shape:", slice_data.shape)
        print("Min Value:", np.min(slice_data))
        print("Max Value:", np.max(slice_data))
        print("Mean Value:", np.mean(slice_data))
        print("Median Value:", np.median(slice_data))
        print("Std Deviation:", np.std(slice_data))
        
        # Check for zero or near-zero values
        zero_percentage = (slice_data == 0).sum() / slice_data.size * 100
        print(f"Percentage of Zero Values: {zero_percentage:.2f}%")
        
        # Check for NaN or Infinite values
        nan_count = np.isnan(slice_data).sum()
        inf_count = np.isinf(slice_data).sum()
        print(f"NaN Values: {nan_count}")
        print(f"Infinite Values: {inf_count}")
        
        # Basic statistical checks
        is_effectively_empty = (
            zero_percentage > 99.9 or  # Almost all zeros
            (np.max(slice_data) - np.min(slice_data)) < 1e-10  # Extremely small variation
        )
        print("Is Effectively Empty:", is_effectively_empty)

    def comprehensive_data_validation(Data_CT, Data_CBF, Data_Tmax, Data_CBV, Data_MTT):

     modalities = [
        ('CT', Data_CT),
        ('CBF', Data_CBF),
        ('Tmax', Data_Tmax),
        ('CBV', Data_CBV),
        ('MTT', Data_MTT)
    ]
    
     for name, data in modalities:
        detailed_slice_diagnosis(data, name)

    def diagnose_slice(slice_data, slice_idx, stage_name):

     print(f"Slice {slice_idx} at {stage_name}:")
     print(f"Min: {np.min(slice_data)}, Max: {np.max(slice_data)}")
     print(f"Mean: {np.mean(slice_data)}, Std: {np.std(slice_data)}")
     print(f"Zero pixels: {np.sum(slice_data == 0)}/{slice_data.size}")

    def extract_metrics_from_text_file(file_path):

     metrics = {}
     current_modality = None
    
     with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Detect modality section
            if line.endswith('Metrics:'):
                current_modality = line.split()[0]
                metrics[current_modality] = {}
            
            # Extract metrics
            if current_modality and "Average MSE:" in line:
                metrics[current_modality]['mse'] = float(line.split(":")[-1].strip())
            if current_modality and "Average SSIM:" in line:
                metrics[current_modality]['ssim'] = float(line.split(":")[-1].strip())
    
     return metrics
    
    def add_metrics_visualization_to_patient_processing(self, patient_idx, patient_output_dir, template_data, Data_CBF, Data_Tmax, Data_CBV, Data_MTT, registered_modalities):

     self.log("Creating directories for visualizations...")
     subject_viz_dir = os.path.join(patient_output_dir, 'visualizations')
     difference_maps_dir = os.path.join(subject_viz_dir, 'difference_maps')
     summary_dir = os.path.join(subject_viz_dir, 'summary')
     slice_viz_dir = os.path.join(subject_viz_dir, 'slice_visualizations')
    
    # Create directories if they don't exist
     for dir_path in [subject_viz_dir, difference_maps_dir, summary_dir, slice_viz_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Create metrics file
     metrics_output_path = os.path.join(patient_output_dir, 'registration_metrics.txt')
    
    # Process metrics and visualizations for each modality
     self.log("Generating metrics and visualizations...")
     with open(metrics_output_path, 'w') as metrics_file:
        metrics_file.write(f"Patient {patient_idx+1} - Registration Metrics\n")
        metrics_file.write("="*50 + "\n\n")
        
        all_modality_metrics = {}
        
        for modality_name, registered_modality in registered_modalities.items():
            # Get original data based on modality name
            original_modality_data = None
            if modality_name == 'CBF':
                original_modality_data = Data_CBF
            elif modality_name == 'Tmax':
                original_modality_data = Data_Tmax
            elif modality_name == 'CBV':
                original_modality_data = Data_CBV
            elif modality_name == 'MTT':
                original_modality_data = Data_MTT
                
            if original_modality_data is None:
                self.log(f"Skipping {modality_name} - original data not found")
                continue
            
            self.log(f"Processing {modality_name} metrics and visualizations...")
            metrics_file.write(f"\n{modality_name} Metrics:\n")
            metrics_file.write("-"*20 + "\n")
            
            # Initialize lists to store slice metrics
            all_mse = []
            all_ssim = []
            slice_indices = []
            
            # Create lists for montage
            template_slices = []
            original_slices = []
            registered_slices = []
            
            # Process each slice
            for slice_idx in range(min(len(original_modality_data), len(template_data))):
                template_slice = template_data[slice_idx]
                original_slice = original_modality_data[slice_idx]
                registered_slice = registered_modality[slice_idx]
                
                if np.all(template_slice == 0) or np.all(original_slice == 0):
                    self.log(f"Skipping empty slice {slice_idx}")
                    continue
                
                # Save slices for montage
                template_slices.append(template_slice)
                original_slices.append(original_slice)
                registered_slices.append(registered_slice)
                
                # Calculate metrics for this slice

                slice_mse, slice_ssim = self.calculate_slice_metrics(template_slice, registered_slice)
                all_mse.append(slice_mse)
                all_ssim.append(slice_ssim)
                slice_indices.append(slice_idx)
                
                slice_metrics = {
                    "mse": slice_mse,
                    "ssim": slice_ssim
                }
                
                # Save visualization for this slice
                png_path = os.path.join(slice_viz_dir, f'{modality_name}_slice_{slice_idx:03d}.png')
                self.save_comparison_image(
                    template_slice,
                    registered_slice, 
                    original_slice,
                    slice_metrics,
                    png_path,
                    modality_name
                )
                
                # Create difference map
                diff_map_path = os.path.join(difference_maps_dir, f'{modality_name}_diff_map_slice_{slice_idx:03d}.png')
                self.create_difference_map(original_slice, registered_slice, diff_map_path)
                
                # Print progress indicator
                if slice_idx % 5 == 0:
                    self.log(f"Processed {slice_idx+1}/{len(original_modality_data)} slices for {modality_name}")
            
            # Store metrics for summary visualization
            modality_metrics = {
                'mse': all_mse,
                'ssim': all_ssim
            }
            all_modality_metrics[modality_name] = modality_metrics
            
            # Create summary visualization
            summary_path = os.path.join(summary_dir, f'{modality_name}_metrics_summary.png')
            self.create_summary_visualization(patient_idx, modality_name, modality_metrics, slice_indices, summary_path)
            
            # Calculate average metrics across all slices
            if len(all_mse) > 0:
                avg_mse = np.mean(all_mse)
                avg_ssim = np.mean(all_ssim)
            else:
                avg_mse = float('nan')
                avg_ssim = float('nan')
            
            # Write metrics to file
            metrics_file.write(f"Average MSE: {avg_mse:.6f}\n")
            metrics_file.write(f"Average SSIM: {avg_ssim:.4f}\n")
            if len(all_mse) > 0:
                metrics_file.write(f"Per-slice SSIM range: {min(all_ssim):.4f} - {max(all_ssim):.4f}\n")
            metrics_file.write(f"Number of valid slices: {len(all_mse)}\n")
            
            # Create montage for overview
            montage_path = os.path.join(summary_dir, f'{modality_name}_montage.png')
            self.create_registration_montage(template_slices, original_slices, registered_slices, modality_name, montage_path)
            
            # Create alignment summary
            alignment_summary_path = os.path.join(summary_dir, f'{modality_name}_alignment_summary.png')
            self.create_alignment_summary(
                template_data,
                original_modality_data,
                registered_modality,
                modality_name,
                alignment_summary_path
            )
        
        # Create combined metrics visualization
        if len(all_modality_metrics) > 0:
            combined_metrics_path = os.path.join(summary_dir, f'all_modalities_comparison.png')
            
            plt.figure(figsize=(15, 10))
            
            # Plot MSE comparison
            plt.subplot(211)
            for modality in all_modality_metrics:
                if len(all_modality_metrics[modality]['mse']) > 0:
                    plt.plot(all_modality_metrics[modality]['mse'], label=modality)
            plt.title(f'MSE Comparison Across Modalities - Patient {patient_idx+1}', fontsize=14)
            plt.xlabel('Slice Index')
            plt.ylabel('MSE (lower is better)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Plot SSIM comparison
            plt.subplot(212)
            for modality in all_modality_metrics:
                if len(all_modality_metrics[modality]['ssim']) > 0:
                    plt.plot(all_modality_metrics[modality]['ssim'], label=modality)
            plt.title(f'SSIM Comparison Across Modalities - Patient {patient_idx+1}', fontsize=14)
            plt.xlabel('Slice Index')
            plt.ylabel('SSIM (higher is better)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(combined_metrics_path, dpi=300, bbox_inches='tight')
            plt.close()
        
     self.log(f"Saved metrics and visualizations for patient {patient_idx+1}")

    
    
    
    
    
    
    
    def rotation_finding(self, data):
        """
        Find orientation and centroid of brain in MRI data
        
        Args:
            data: Input MRI data
            
        Returns:
            Tuple of orientation and centroid coordinates
        """
        seedd = data.max()
        data_new = []
        area = []
        for i in range(data.shape[0]):
            image = (data[i] > data.min())
            seed = np.copy(image)
            seed[1:-1, 1:-1] = 1.0
            mask = image
            data_new.append(reconstruction(seed, mask, method='erosion'))
            area.append(np.sum(data_new[i]))
        area = np.stack(area)
        area_idx = np.argmax(area)
        area_data = data_new[area_idx]
        all_labels = measure.label(area_data, background=0)
        properties = measure.regionprops(all_labels)
        area = [prop.area for prop in properties]
        area = np.stack(area)
        area_idx = np.argmax(area)
        y0, x0 = properties[area_idx].centroid
        orientation = properties[area_idx].orientation
        return orientation, [y0, x0]

    def skull_stripping(self, mask_data, orient):
        """
        Remove skull from brain MRI data and extract region properties
        
        Args:
            mask_data: Input mask data
            orient: Orientation data
            
        Returns:
            Dictionary of region properties
        """
        region_prop = dict()
        for i in range(mask_data.shape[0]):
            image = mask_data[i]
            seed = np.copy(image)
            seed[1:-1, 1:-1] = image.max()
            mask = image
            mask_data[i] = reconstruction(seed, mask, method='erosion')
        for i in range(mask_data.shape[0]):
            all_labels = measure.label(mask_data[i], background=0)
            properties = measure.regionprops(all_labels)
            if len(properties) == 0:
                continue
            else:
                area = [prop.area for prop in properties]
                area = np.stack(area)
                area_idx = np.argmax(area)
                bbox = properties[area_idx].bbox
                filled_image = properties[area_idx].filled_image
                if filled_image.shape[0] > 64 and filled_image.shape[1] > 64:
                    region_prop[i] = {'bbox': bbox, 'filled_image': filled_image, 'orientation': orient[0], 'centroid': orient[1]}
        return region_prop

    def cropping_data(self, data, region_prop, normli=False, rot=False):
        """
        Crop data using region properties
        
        Args:
            data: Input data
            region_prop: Region properties dictionary
            normli: Whether to normalize
            rot: Whether to rotate
            
        Returns:
            List of cropped data
        """
        data_list = []
        for i in region_prop.keys():
            bbox = region_prop[i]['bbox']
            filled_image = region_prop[i]['filled_image']
            orien = region_prop[i]['orientation']
            centeroid = region_prop[i]['centroid']
            data_temp = data[i, bbox[0]:bbox[2], bbox[1]:bbox[3]] * filled_image
            if rot:
                data_temp = transform.rotate(data_temp, angle=-orien*180/3.14, center=(centeroid[0]-bbox[0], centeroid[1]-bbox[1]))
                filled_image = transform.rotate(filled_image, angle=-orien*180/3.14, center=(centeroid[0]-bbox[0], centeroid[1]-bbox[1]))
            if normli:
                filled_image = np.round(filled_image)
                filled_idx = np.argwhere(filled_image > 0)
                data_temp = (data_temp - np.mean(data_temp[filled_idx[:, 0], filled_idx[:, 1]])) / (np.std(data_temp[filled_idx[:, 0], filled_idx[:, 1]]) + 1)
            data_list.append(data_temp)
        return data_list

    def data_normalization(self, data):
        """
        Normalize each slice in the data
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        for j in range(len(data)):
            m = np.mean(data[j])
            s = np.std(data[j])
            data[j] = (data[j] - m) / (s + 1)
        return data

    def patch_generation_forward(self, data_CT, data_CBF, data_Tmax, data_CBV, data_MTT, ii_jj_start=[0, 4, 8, 12]):
        """
        Generate forward patches for registration from CT and perfusion data
        
        Args:
            data_CT: CT data
            data_CBF: CBF data
            data_Tmax: Tmax data
            data_CBV: CBV data
            data_MTT: MTT data
            ii_jj_start: Starting indices
            
        Returns:
            Tuple of fixed patches, moving patches, and indices
        """
        fixed_patches = []
        moving_patches = []
        indices = []
        
        for j in range(len(data_CT)):
            ct = data_CT[j]
            cbf = data_CBF[j]
            tmax = data_Tmax[j]
            cbv = data_CBV[j]
            mtt = data_MTT[j]
            
            min_shape = np.min([arr.shape for arr in [ct, cbf, tmax, cbv, mtt]], axis=0)
            ct = ct[:min_shape[0], :min_shape[1]]
            cbf = cbf[:min_shape[0], :min_shape[1]]
            tmax = tmax[:min_shape[0], :min_shape[1]]
            cbv = cbv[:min_shape[0], :min_shape[1]]
            mtt = mtt[:min_shape[0], :min_shape[1]]
            
            for ii_jj in ii_jj_start:
                ii = ii_jj
                inc_ii = 16
                inc_jj = 16
                while ii < ct.shape[0]:
                    if ii + 64 >= ct.shape[0]:
                        ii = ct.shape[0] - 64
                        inc_ii = 64
                    else:
                        inc_ii = 16
                    jj = ii_jj
                    while jj < ct.shape[1]:
                        if jj + 64 >= ct.shape[1]:
                            jj = ct.shape[1] - 64
                            inc_jj = 64
                        else:
                            inc_jj = 16
                        fixed_patch = ct[ii:ii+64, jj:jj+64]
                        moving_patches_temp = np.stack([
                            cbf[ii:ii+64, jj:jj+64],
                            tmax[ii:ii+64, jj:jj+64],
                            cbv[ii:ii+64, jj:jj+64],
                            mtt[ii:ii+64, jj:jj+64]
                        ], axis=-1)
                        fixed_patches.append(fixed_patch)
                        moving_patches.append(moving_patches_temp)
                        indices.append([j, ii, jj])
                        jj += inc_jj
                    ii += inc_ii
        
        fixed_patches = np.expand_dims(np.array(fixed_patches), axis=-1)
        moving_patches = np.array(moving_patches)
        indices = np.array(indices)
        
        return fixed_patches, moving_patches, indices

    def patch_generation_sym(self, data_CT, data_CBF, data_Tmax, data_CBV, data_MTT, ii_jj_start=[0, 4, 8, 12]):
        """
        Generate symmetric patches by flipping the images horizontally
        
        Args:
            data_CT: CT data
            data_CBF: CBF data
            data_Tmax: Tmax data
            data_CBV: CBV data
            data_MTT: MTT data
            ii_jj_start: Starting indices
            
        Returns:
            Tuple of fixed patches, moving patches, and indices
        """
        fixed_patches = []
        moving_patches = []
        indices = []
        
        for j in range(len(data_CT)):
            ct = data_CT[j][:, ::-1]  # Horizontal flip
            cbf = data_CBF[j][:, ::-1]
            tmax = data_Tmax[j][:, ::-1]
            cbv = data_CBV[j][:, ::-1]
            mtt = data_MTT[j][:, ::-1]
            
            min_shape = np.min([arr.shape for arr in [ct, cbf, tmax, cbv, mtt]], axis=0)
            ct = ct[:min_shape[0], :min_shape[1]]
            cbf = cbf[:min_shape[0], :min_shape[1]]
            tmax = tmax[:min_shape[0], :min_shape[1]]
            cbv = cbv[:min_shape[0], :min_shape[1]]
            mtt = mtt[:min_shape[0], :min_shape[1]]
            
            for ii_jj in ii_jj_start:
                ii = ii_jj
                inc_ii = 16
                inc_jj = 16
                while ii < ct.shape[0]:
                    if ii + 64 >= ct.shape[0]:
                        ii = ct.shape[0] - 64
                        inc_ii = 64
                    else:
                        inc_ii = 16
                    jj = ii_jj
                    while jj < ct.shape[1]:
                        if jj + 64 >= ct.shape[1]:
                            jj = ct.shape[1] - 64
                            inc_jj = 64
                        else:
                            inc_jj = 16
                        fixed_patch = ct[ii:ii+64, jj:jj+64]
                        moving_patches_temp = np.stack([
                            cbf[ii:ii+64, jj:jj+64],
                            tmax[ii:ii+64, jj:jj+64],
                            cbv[ii:ii+64, jj:jj+64],
                            mtt[ii:ii+64, jj:jj+64]
                        ], axis=-1)
                        fixed_patches.append(fixed_patch)
                        moving_patches.append(moving_patches_temp)
                        indices.append([j, ii, jj])
                        jj += inc_jj
                    ii += inc_ii
        
        fixed_patches = np.expand_dims(np.array(fixed_patches), axis=-1)
        moving_patches = np.array(moving_patches)
        indices = np.array(indices)
        
        return fixed_patches, moving_patches, indices
        
    def reconstruct_deformable_field(self, deformable_patches, indices, original_shape):
        """
        Reconstruct full deformation field from patches
        
        Args:
            deformable_patches: Deformation patches
            indices: Patch indices
            original_shape: Original shape to reconstruct
            
        Returns:
            Reconstructed deformation field
        """
        result = np.zeros(original_shape + (deformable_patches.shape[-1],), dtype=np.float32)
        count = np.zeros(original_shape[:3], dtype=np.float32)

        for patch, (slice_idx, x, y) in zip(deformable_patches, indices):
            z_start = slice_idx
            z_end = min(z_start + 1, original_shape[0])
            x_end = min(x + patch.shape[0], original_shape[1])
            y_end = min(y + patch.shape[1], original_shape[2])

            patch_z = z_end - z_start
            patch_x = x_end - x
            patch_y = y_end - y

            result[z_start:z_end, x:x_end, y:y_end] += patch[:patch_x, :patch_y]
            count[z_start:z_end, x:x_end, y:y_end] += 1

        # Normalize overlapping regions
        count = np.maximum(count, 1)
        for i in range(result.shape[-1]):
            result[:, :, :, i] /= count

        return result

    def apply_deformation_field(self, image, deformation_field):
        """
        Apply a 2D deformation field to a 3D image using interpolation
        
        Args:
            image: Original image to deform, shape (z, x, y)
            deformation_field: Deformation field to apply, shape (z, x, y, 2)
            
        Returns:
            Deformed image
        """
        if image.shape[:3] != deformation_field.shape[:3]:
            raise ValueError(f"Image shape {image.shape} and deformation field shape {deformation_field.shape} are not compatible")

        # Check if deformation field has only 2 components (x, y)
        if deformation_field.shape[-1] == 2:
            # Create a copy of the image to store the result
            result = np.zeros_like(image)
            
            # Process each slice separately
            for z in range(image.shape[0]):
                # Create base coordinates for this slice
                x_coords, y_coords = np.meshgrid(
                    np.arange(image.shape[1]),
                    np.arange(image.shape[2]),
                    indexing='ij'
                )
                
                # Add deformation and clip to valid bounds for this slice
                x_deformed = x_coords + deformation_field[z, :, :, 0]
                y_deformed = y_coords + deformation_field[z, :, :, 1]
                
                x_deformed = np.clip(x_deformed, 0, image.shape[1] - 1)
                y_deformed = np.clip(y_deformed, 0, image.shape[2] - 1)
                
                # Create coordinates for interpolation
                coords = np.stack([x_deformed, y_deformed], axis=-1)
                
                # Create interpolator for this slice
                from scipy.interpolate import RegularGridInterpolator
                
                # Define grid points
                grid_x = np.arange(image.shape[1])
                grid_y = np.arange(image.shape[2])
                
                # Create interpolator
                interpolator = RegularGridInterpolator(
                    (grid_x, grid_y),
                    image[z],
                    method='linear',
                    bounds_error=False,
                    fill_value=None  # Use nearest neighbor for out-of-bounds
                )
                
                # Reshape coordinates for interpolation
                points = coords.reshape(-1, 2)
                
                # Interpolate and reshape back
                result[z] = interpolator(points).reshape(image[z].shape)
            
            return result
        else:
            raise ValueError(f"Deformation field must have 2 components (x, y), but got {deformation_field.shape[-1]}")

    def create_ensemble_model(self, weights_dir, ensemble_method='weighted_average'):
        """
        Create an ensemble model from multiple weight files
        
        Args:
            weights_dir: Directory containing weight files from different folds
            ensemble_method: Method for combining predictions ('weighted_average' or 'average')
        
        Returns:
            Function that performs ensemble prediction
        """
        # Find all weight files - more flexible pattern matching
        weight_files = glob.glob(os.path.join(weights_dir, '*.h5'))
        
        if not weight_files:
            raise ValueError(f"No weight files found in {weights_dir}")
        
        self.log(f"Creating ensemble with {len(weight_files)} models")
        
        # Sort weight files for consistent ordering
        weight_files = sorted(weight_files)
        
        # Weights for ensemble (can be tuned based on validation performance)
        if ensemble_method == 'weighted_average':
            # Create weights based on the number of files found
            ensemble_weights = []
            
            # Simple decreasing weights strategy: higher weights to earlier models
            total_files = len(weight_files)
            for i in range(total_files):
                weight = 1.0 - (i * 0.5 / total_files)  # Gradual decrease
                ensemble_weights.append(max(0.1, weight))  # Ensure minimum weight
                
            # Normalize weights
            total = sum(ensemble_weights)
            ensemble_weights = [w/total for w in ensemble_weights]
        else:
            # Equal weights for all models
            ensemble_weights = [1.0/len(weight_files)] * len(weight_files)
        
        # Create a function for ensemble prediction
        def ensemble_predict(inputs):
            predictions = []
            
            for i, weight_file in enumerate(weight_files):
                self.log(f"Loading weights from {weight_file}")
                self.model.load_weights(weight_file)
                pred = self.model.predict(inputs, batch_size=1)
                predictions.append(pred)
            
            # Combine predictions with weights
            ensemble_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += pred * ensemble_weights[i]
            
            self.log(f"Ensemble prediction complete using {len(weight_files)} models")
            return ensemble_pred
        
        return ensemble_predict
        
    def preprocess_and_register_patients(self, dataset_path, template_path, output_dir, weights_dir=None, use_ensemble=True):
        """
        Preprocess and register 5 patients
        
        Args:
            dataset_path: Path to dataset directory
            template_path: Path to template image
            output_dir: Path to save results
            weights_dir: Directory containing model weights
            use_ensemble: Whether to use ensemble model
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get list of patient directories
        patient_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])[:5]
        self.log(f"Processing {len(patient_dirs)} patients from {dataset_path}")
        
        # Load template
        self.log(f"Loading template from {template_path}")
        template_nii = nib.load(template_path)
        template_data = template_nii.get_fdata().astype('float32')
        if template_data.ndim == 3:
            template_data = template_data.transpose(2, 0, 1)
        
        # Initialize model if not already created
        if self.model is None:
            self.log("Creating registration model...")
            self.create_registration_model(
                fixed_input_shape=(64, 64, 1),
                moving_input_shape=(64, 64, 4),
                n_base_filters=8,
                depth=4,
                dropout_rate=0.4
            )
        
        # Load weights
        if weights_dir is None:
            weights_dir = '.'  # Use current directory if not specified
        
        # Check for ensemble weights - more flexible pattern matching
        ensemble_predictor = None
        if use_ensemble:
            # Find any .h5 files in the weights directory
            weight_files = glob.glob(os.path.join(weights_dir, '*.h5'))
            
            if len(weight_files) >= 5:
                self.log(f"Found {len(weight_files)} weight files for ensemble at {weights_dir}")
                ensemble_predictor = self.create_ensemble_model(weights_dir)
            else:
                self.log(f"Not enough ensemble weights found in {weights_dir} (found {len(weight_files)}, need 5), using single model")
                use_ensemble = False
                
                # Try to load a single model weight if available
                if weight_files:
                    self.log(f"Loading single model weights from {weight_files[0]}")
                    self.model.load_weights(weight_files[0])
                else:
                    self.log("No model weights found, using randomly initialized model")
        
        # Process each patient
        for patient_idx, patient_dir in enumerate(patient_dirs):
            patient_path = os.path.join(dataset_path, patient_dir)
            patient_output_dir = os.path.join(output_dir, f"patient_{patient_idx+1}")
            os.makedirs(patient_output_dir, exist_ok=True)
            
            self.log(f"\nProcessing patient {patient_idx+1}: {patient_dir}")
            
            # Find and load files for this patient
            try:
                # Find CT and perfusion files
                ct_file = glob.glob(os.path.join(patient_path, '*CT*.nii'))[0]
                cbf_file = glob.glob(os.path.join(patient_path, '*CT_CBF*.nii'))[0]
                cbv_file = glob.glob(os.path.join(patient_path, '*CT_CBV*.nii'))[0]
                mtt_file = glob.glob(os.path.join(patient_path, '*CT_MTT*.nii'))[0]
                tmax_file = glob.glob(os.path.join(patient_path, '*CT_Tmax*.nii'))[0]
                
                self.log(f"Loading CT data from {ct_file}")
                x = nib.load(ct_file)
                Data_CT = x.get_fdata().astype('float32').transpose(2, 0, 1)
                
                # Load and preprocess other modalities
                self.log("Loading and preprocessing other modalities...")
                Data_CBF = nib.load(cbf_file).get_fdata().astype('float32').transpose(2, 0, 1)
                Data_Tmax = nib.load(tmax_file).get_fdata().astype('float32').transpose(2, 0, 1)
                Data_CBV = nib.load(cbv_file).get_fdata().astype('float32').transpose(2, 0, 1)
                Data_MTT = nib.load(mtt_file).get_fdata().astype('float32').transpose(2, 0, 1)
                
                # Preprocessing CT data
                temp = Data_Tmax + Data_CBV + Data_MTT + Data_CBF
                temp1 = (temp > 0) * 1.0
                rot = self.rotation_finding(Data_CT)
                region_prop = self.skull_stripping(temp1, rot)
                
                Temp_CT = self.cropping_data(Data_CT, region_prop, normli=False, rot=False)
                Temp_CT = self.data_normalization(Temp_CT)
                
                Temp_CBF = self.cropping_data(Data_CBF, region_prop, normli=False, rot=False)
                Temp_CBF = self.data_normalization(Temp_CBF)
                
                Temp_Tmax = self.cropping_data(Data_Tmax, region_prop, normli=False, rot=False)
                Temp_Tmax = self.data_normalization(Temp_Tmax)
                
                Temp_CBV = self.cropping_data(Data_CBV, region_prop, normli=False, rot=False)
                Temp_CBV = self.data_normalization(Temp_CBV)
                
                Temp_MTT = self.cropping_data(Data_MTT, region_prop, normli=False, rot=False)
                Temp_MTT = self.data_normalization(Temp_MTT)
                
                # Process template with resizing
                from skimage.transform import resize
                
                # Resize template to match CT dimensions
                processed_template = []
                for i in range(len(Temp_CT)):
                    template_slice = template_data[i % template_data.shape[0]]
                    if template_slice.shape != Temp_CT[i].shape:
                        template_slice = resize(template_slice, Temp_CT[i].shape, order=1, mode='constant', anti_aliasing=True)
                    processed_template.append(template_slice)
                
                Temp_Template = processed_template
                
                # Generate patches
                self.log("Generating forward patches...")
                forward_patches, forward_moving_patches, forward_indices = self.patch_generation_forward(
                    Temp_Template, Temp_CBF, Temp_Tmax, Temp_CBV, Temp_MTT)
                
                self.log("Generating symmetric patches...")
                symmetric_patches, symmetric_moving_patches, symmetric_indices = self.patch_generation_sym(
                    Temp_Template, Temp_CBF, Temp_Tmax, Temp_CBV, Temp_MTT)
                
                # Perform registration prediction
                if use_ensemble and ensemble_predictor is not None:
                    self.log("Performing ensemble registration prediction...")
                    predicted_patches_forward = ensemble_predictor([forward_patches, forward_moving_patches])
                    predicted_patches_symmetric = ensemble_predictor([symmetric_patches, symmetric_moving_patches])
                else:
                    self.log("Performing single model registration prediction...")
                    predicted_patches_forward = self.model.predict([forward_patches, forward_moving_patches], batch_size=1)
                    predicted_patches_symmetric = self.model.predict([symmetric_patches, symmetric_moving_patches], batch_size=1)
                
                # Extract deformation field from predictions
                deformable_field_patches = predicted_patches_symmetric[:, :, :, -2:]
                self.log("Reconstructing deformable field...")
                full_deformable_field = self.reconstruct_deformable_field(deformable_field_patches, symmetric_indices, Data_CT.shape)
                
                # Save deformation field
                deformable_img = nib.Nifti1Image(full_deformable_field, x.affine, x.header)
                deformable_output_path = os.path.join(patient_output_dir, 'deformation_field.nii.gz')
                nib.save(deformable_img, deformable_output_path)
                self.log(f"Saved deformation field at {deformable_output_path}")
                # Apply deformation to each modality
                self.log("Applying deformation field to all modalities...")
                registered_modalities = {}

                for modality_name, modality_data in [('CBF', Data_CBF), ('Tmax', Data_Tmax), ('CBV', Data_CBV), ('MTT', Data_MTT)]:
                  registered_modality = self.apply_deformation_field(modality_data, full_deformable_field)
                  registered_modalities[modality_name] = registered_modality
                  reg_img = nib.Nifti1Image(registered_modality, x.affine, x.header)
                  reg_output_path = os.path.join(patient_output_dir, f'registered_{modality_name}.nii.gz')
                  nib.save(reg_img, reg_output_path)
                  self.log(f"Saved registered {modality_name} at {reg_output_path}")


                self.add_metrics_visualization_to_patient_processing(
                 patient_idx,
                 patient_output_dir,
                 template_data,
                 Data_CBF, Data_Tmax, Data_CBV, Data_MTT,
                 registered_modalities)

            
            except Exception as e:
                self.log(f"Error processing patient {patient_dir}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Save log
        log_path = os.path.join(output_dir, 'registration_log.txt')
        self.save_log(log_path)
        self.log(f"Registration completed for all patients. Log saved to {log_path}")
        
        return True

# Add command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain Template Registration Tool v{}'.format(__version__))
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--template', type=str, required=True, help='Path to template image file')
    parser.add_argument('--output', type=str, required=True, help='Path to output directory')
    parser.add_argument('--weights', type=str, default=None, help='Directory containing model weights')
    parser.add_argument('--use-ensemble', action='store_true', help='Use ensemble model if multiple weights are available')
    parser.add_argument('--gpu', type=str, default="0", help='GPU device ID')
    
    args = parser.parse_args()
    
    # Create registration object
    registration = BrainRegistration(gpu_id=args.gpu)
    
    # Run registration
    registration.preprocess_and_register_patients(
        dataset_path=args.dataset,
        template_path=args.template,
        output_dir=args.output,
        weights_dir=args.weights,
        use_ensemble=args.use_ensemble
    )
