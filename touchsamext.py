from TDStoreTools import StorageManager
import TDFunctions as TDF

from enum import Enum
from datetime import datetime
import webbrowser
import random
import cv2
import numpy as np

# Set a fixed seed for reproducibility
random.seed(42)

try:
    import torch
    from ultralytics import FastSAM
except Exception as e:
    import traceback
    print(traceback.format_exc())
    print(f"Import error: {e}")
    current_time = datetime.now()
    formated_time = current_time.strftime("%H:%M:%S")
    op('fifo1').appendRow([formated_time, 'Error', e])


class SegmentationMode(Enum):
    ANYTHING = 0
    POINTS = 1
    TEXT = 2


class FastSAMResultsManager:
    """
    Manages and exposes FastSAM segmentation results in TouchDesigner
    """
    def __init__(self):
        # Store raw results from the model
        self.raw_results = None
        
        # Structured results for easy access
        self.num_objects = 0
        self.mask_data = []  # List of dicts containing mask info
        
        # For DAT export
        self.results_table = None
        
        # Define headers once
        self.headers = ['id', 'confidence', 'center_x', 'center_y', 
                        'x1', 'y1', 'x2', 'y2', 'width', 'height', 'area']
        
    def process_results(self, results):
        """
        Process raw results from FastSAM model and structure them
        """
        self.raw_results = results
        self.mask_data = []
        
        if not results or len(results) == 0:
            self.num_objects = 0
            return
            
        try:
            # Extract masks and their associated data
            masks = results[0].masks
            boxes = results[0].boxes
            self.num_objects = len(masks)
            
            # Pre-allocate mask_data for better performance
            self.mask_data = [{} for _ in range(self.num_objects)]
            
            # Get all boxes data at once for better performance
            if boxes:
                all_boxes = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else None
                all_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            
            # Get mask data as numpy arrays (batch processing)
            all_masks = masks.data.cpu().numpy()
            
            # Process each detection
            for i in range(self.num_objects):
                mask_info = self.mask_data[i]
                
                # Mask numpy version (avoid tensor conversion for each mask)
                mask_info['mask'] = all_masks[i].squeeze()
                
                # Get bounding box if available
                if all_boxes is not None and i < len(all_boxes):
                    bbox = all_boxes[i]
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    mask_info['bbox'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    mask_info['width'] = x2 - x1
                    mask_info['height'] = y2 - y1
                    mask_info['center_x'] = (x1 + x2) / 2
                    mask_info['center_y'] = (y1 + y2) / 2
                    mask_info['area'] = mask_info['width'] * mask_info['height']
                
                # Get confidence if available
                if all_conf is not None and i < len(all_conf):
                    mask_info['confidence'] = float(all_conf[i])
                
                # Add default tracking ID
                mask_info['track_id'] = i + 1
            
            # Sort by area (largest first) for consistency
            self.mask_data.sort(key=lambda x: x.get('area', 0), reverse=True)
            
            # Create a table representation for DAT export
            self._create_results_table()
            
        except Exception as e:
            import traceback
            print(f"Error processing FastSAM results: {e}")
            print(traceback.format_exc())
            self.num_objects = 0
    
    def _create_results_table(self):
        """
        Create a table representation of results for DAT export
        """
        # Initialize with header row
        table_data = [self.headers]
        
        # Add data for each detection
        for i, mask in enumerate(self.mask_data):
            # Direct access to values with default fallbacks (more efficient)
            bbox = mask.get('bbox', {})
            row = [
                mask.get('track_id', i+1),
                round(mask.get('confidence', 0), 3),
                round(mask.get('center_x', 0), 1),
                round(mask.get('center_y', 0), 1),
                round(bbox.get('x1', 0), 1),
                round(bbox.get('y1', 0), 1),
                round(bbox.get('x2', 0), 1),
                round(bbox.get('y2', 0), 1),
                round(mask.get('width', 0), 1),
                round(mask.get('height', 0), 1),
                round(mask.get('area', 0), 1)
            ]
            table_data.append(row)
            
        self.results_table = table_data
        
    def export_to_dat(self, dat_op):
        """
        Export results to a TouchDesigner Table DAT
        """
        # Clear the existing DAT
        dat_op.clear()
        
        if self.results_table:
            # Append rows as a single batch operation
            dat_op.appendRows(self.results_table)
        else:
            # Just add the headers
            dat_op.appendRow(self.headers)
                
    def get_mask_by_id(self, track_id):
        """
        Return mask data for a specific tracking ID
        """
        # Use a list comprehension with next() for more efficient lookup
        try:
            return next(mask for mask in self.mask_data if mask.get('track_id') == track_id)
        except StopIteration:
            return None
    
    def get_largest_mask(self):
        """
        Return data for the largest mask by area
        """
        if not self.mask_data:
            return None
        return max(self.mask_data, key=lambda x: x.get('area', 0))
        
    def get_most_central_mask(self, image_width, image_height):
        """
        Return mask closest to center of image
        """
        if not self.mask_data:
            return None
            
        center_x = image_width / 2
        center_y = image_height / 2
        
        # Pre-calculate squared distances for better performance
        return min(
            self.mask_data, 
            key=lambda mask: (mask.get('center_x', 0) - center_x)**2 + (mask.get('center_y', 0) - center_y)**2
        )


class TouchSamExt:
    """
    FastSAM segmentation extension for TouchDesigner
    """
    def __init__(self, ownerComp):
        # The component to which this extension is attached
        self.ownerComp = ownerComp
        
        # Get parameters from the parameter component
        params = op("parameter1")
        self.model_name = params["Model", 1].val
        self.loaded = params["Loadmodel", 1].val
        self.image_size = int(params["Imagesize", 1].val)
        self.device = params["Device", 1].val
        self.confidence = float(params["Confidence", 1].val)
        self.iou = float(params["Iou", 1].val)
        
        # Create results manager
        self.results_manager = FastSAMResultsManager()
                
        # Create references to output DATs
        self.results_dat = op('results_dat')
        
        self.segmentation_mode = SegmentationMode.ANYTHING  # Default mode
        self.points = None
        self.prompt = None
        
        # Store binary masks
        self.masks = []
        
        # Pre-defined colors for masks (stored as numpy array for faster access)
        self.fixed_colors = np.array([
            [255, 0, 0],      # Red
            [0, 255, 0],      # Green
            [0, 0, 255],      # Blue
            [255, 255, 0],    # Yellow
            [255, 0, 255],    # Magenta
            [0, 255, 255],    # Cyan
            [128, 0, 0],      # Maroon
            [0, 128, 0],      # Green (dark)
            [0, 0, 128],      # Navy
            [128, 128, 0],    # Olive
            [128, 0, 128],    # Purple
            [0, 128, 128],    # Teal
            [255, 165, 0],    # Orange
            [255, 192, 203],  # Pink
            [165, 42, 42]     # Brown
        ], dtype=np.uint8)
        
        self.input_image = op('input_image')
        self.model = None
        
        # For tracking masks across frames
        self.previous_centroids = {}  # Store centroids of previous masks
        self.next_id = 1              # Next available ID
        
        # Tracking constants
        self.max_dist_threshold_ratio = 0.1  # Distance threshold as percentage of image diagonal

    def load_model(self):
        try:
            # Create a FastSAM model
            self.model = FastSAM(f"{self.model_name}.pt")
            device = "GPU" if self.device == "cuda:0" else "CPU"
            message = f"Successfully loaded {self.model_name} on {device}"
            print(message)
            self.fifolog("Status", message)
        except Exception as e:
            import traceback
            print(f"Error loading model: {e}")
            print(traceback.format_exc())
            self.fifolog("Error", traceback.format_exc())

    def get_mask_color(self, mask_id):
        """
        Get a consistent color for a mask based on its ID (vectorized)
        """
        # Use modulo to cycle through available colors
        color_idx = (mask_id - 1) % len(self.fixed_colors)
        return self.fixed_colors[color_idx]

    def get_mask_centroid(self, mask):
        """
        Calculate the centroid of a binary mask
        """
        # Find non-zero pixels efficiently using numpy
        indices = np.where(mask > 0.5)
        
        # Check if mask has any non-zero pixels
        if len(indices[0]) == 0:
            return None
            
        # Calculate centroid
        return np.mean(indices[1]), np.mean(indices[0])  # x, y

    def track_masks(self, current_masks, img_width, img_height):
        """
        Track masks by centroid distance between frames
        Returns assigned IDs for current masks
        """
        # Calculate maximum distance threshold based on image size
        max_dist_threshold = np.sqrt(img_width**2 + img_height**2) * self.max_dist_threshold_ratio
        
        # Calculate centroids for current masks
        current_centroids = []
        for mask in current_masks:
            centroid = self.get_mask_centroid(mask)
            current_centroids.append(centroid if centroid is not None else (-1, -1))
        
        # First frame or no previous masks
        if not self.previous_centroids:
            self.previous_centroids = {i+1: centroid for i, centroid in enumerate(current_centroids)}
            return list(range(1, len(current_masks) + 1))
        
        # Prepare arrays for assignment
        assignments = []
        used_prev_ids = set()
        
        # For each current centroid, find the best match
        for curr_idx, curr_centroid in enumerate(current_centroids):
            if curr_centroid[0] < 0:  # Invalid centroid
                new_id = self.next_id
                self.next_id += 1
                assignments.append(new_id)
                continue
                
            best_dist = max_dist_threshold
            best_prev_id = None
            
            # Find closest previous centroid
            for prev_id, prev_centroid in self.previous_centroids.items():
                if prev_id in used_prev_ids or prev_centroid[0] < 0:
                    continue
                    
                # Calculate Euclidean distance
                dist = np.sqrt((curr_centroid[0] - prev_centroid[0])**2 + 
                              (curr_centroid[1] - prev_centroid[1])**2)
                
                if dist < best_dist:
                    best_dist = dist
                    best_prev_id = prev_id
            
            if best_prev_id is not None:
                # Found a match
                assignments.append(best_prev_id)
                used_prev_ids.add(best_prev_id)
            else:
                # No match found, assign new ID
                new_id = self.next_id
                self.next_id += 1
                assignments.append(new_id)
        
        # Update previous centroids for next frame
        self.previous_centroids = {assignments[i]: centroid 
                                  for i, centroid in enumerate(current_centroids)
                                  if centroid[0] >= 0}
        
        return assignments
    
    def annotate_image(self, original_image, binary_masks):
        """
        Overlays binary masks onto the original image with consistent colors
        """
        # Make a copy to avoid modifying original
        annotated_image = original_image.copy()
        
        # Convert image format if needed
        if annotated_image.shape[-1] == 4:  # BGRA
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGRA2BGR)
        
        # No masks to process
        if not binary_masks or len(binary_masks) == 0:
            self.masks = []
            return annotated_image
            
        # Track masks across frames
        mask_ids = self.track_masks(binary_masks, annotated_image.shape[1], annotated_image.shape[0])
        
        # Store binary masks (deep copy)
        self.masks = [mask.copy() for mask in binary_masks]
        
        # Create a single overlay for all masks (more efficient)
        overlay = np.zeros_like(annotated_image)
        
        # Apply all masks at once
        for i, (mask, mask_id) in enumerate(zip(binary_masks, mask_ids)):
            # Get color for this mask
            color = self.get_mask_color(mask_id)
            
            # Create a boolean mask
            mask_bool = mask.squeeze() > 0.5
            
            # Apply color to overlay where mask is active
            overlay[mask_bool] = color
        
        # Blend the overlay with the original image (in-place for efficiency)
        alpha = 0.5
        cv2.addWeighted(annotated_image, 1 - alpha, overlay, alpha, 0, annotated_image)
        
        return annotated_image

    def segment_image(self, scriptOp):
        """
        Main segmentation function called by TouchDesigner
        """
        try:
            # Get the input image
            input_image = self.input_image.numpyArray(delayed=False)
            
            if input_image is None or input_image.size == 0:
                print("Error: Input image is None or empty")
                return
            
            # Store original dimensions
            original_height, original_width = input_image.shape[:2]
            
            # Convert to RGB if needed (for model input)
            if input_image.shape[-1] == 4:  # BGRA
                rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGRA2RGB)
            elif input_image.shape[-1] == 3:  # BGR
                rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = input_image
                
            # Create a copy for processing 
            processing_image = rgb_image
            
            # Skip resizing if not needed
            if self.image_size > 0:
                # Calculate dimensions while preserving aspect ratio
                scale_factor = min(self.image_size / original_width, self.image_size / original_height)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                
                # Only resize if necessary
                if new_width != original_width or new_height != original_height:
                    processing_image = cv2.resize(rgb_image, (new_width, new_height), 
                                                 interpolation=cv2.INTER_LINEAR)
                
            # Ensure processing image is in the correct format
            if processing_image.dtype != np.uint8:
                processing_image = (processing_image * 255).astype(np.uint8)

            # Run model inference
            if self.model is not None:
                #with torch.no_grad():  # Disable gradient computation for inference
                # Run the model
                results = self.model.track(
                    source=processing_image,
                    texts=self.prompt,
                    points=self.points,
                    device=self.device,
                    imgsz=max(processing_image.shape[1], processing_image.shape[0]),
                    conf=self.confidence,
                    iou=self.iou,
                    retina_masks=False,
                    stream=False,
                    verbose=False
                )
                
                # Process results with our manager
                self.results_manager.process_results(results)
                
                # Export results to DAT if available
                if self.results_dat:
                    self.results_manager.export_to_dat(self.results_dat)
                
                # Extract masks for visualization
                binary_masks = []
                if self.results_manager.num_objects > 0:
                    # Process all masks at once where possible
                    for mask_info in self.results_manager.mask_data:
                        mask_np = mask_info['mask']
                        
                        # Resize to original dimensions if needed
                        if mask_np.shape[0] != original_height or mask_np.shape[1] != original_width:
                            mask_np = cv2.resize(mask_np, (original_width, original_height), 
                                                interpolation=cv2.INTER_NEAREST)
                        
                        # Convert to binary mask
                        mask_binary = (mask_np > 0.5).astype(np.float32)
                        binary_masks.append(np.expand_dims(mask_binary, axis=-1))
                
                # Track masks and apply visualization
                if binary_masks:
                    # Get mask IDs from tracking
                    mask_ids = self.track_masks(binary_masks, original_width, original_height)
                    
                    # Update the tracking IDs in the result manager
                    for i, mask_id in enumerate(mask_ids):
                        if i < len(self.results_manager.mask_data):
                            self.results_manager.mask_data[i]['track_id'] = mask_id
                    
                    # Re-export with updated tracking IDs
                    if self.results_dat:
                        self.results_manager.export_to_dat(self.results_dat)
                    
                    # Annotate the image with masks
                    annotated_image = self.annotate_image(input_image, binary_masks)
                    scriptOp.copyNumpyArray(annotated_image)
                else:
                    # No masks detected, return original
                    scriptOp.copyNumpyArray(input_image)
            else:
                print("Model not loaded")
                self.fifolog("Status", "Model not loaded")
                scriptOp.copyNumpyArray(input_image)  # Return original if no model
        except Exception as e:
            import traceback
            print(f"Error in segment_image: {e}")
            print(traceback.format_exc())
            self.fifolog("Error", traceback.format_exc())
            # Try to return the original image if possible
            try:
                if 'input_image' in locals() and input_image is not None:
                    scriptOp.copyNumpyArray(input_image)
            except:
                pass  # Last resort if even returning the original fails
    
    def par_exec_onValueChange(self, par):
        """
        Handle parameter changes
        """
        param_name = par.name
        param_value = par.eval()
        
        # Update parameters based on name (more efficient than multiple if-else)
        param_handlers = {
            "Device": lambda v: setattr(self, 'device', v),
            "Imagesize": lambda v: setattr(self, 'image_size', int(v)),
            "Confidence": lambda v: setattr(self, 'confidence', float(v)),
            "Iou": lambda v: setattr(self, 'iou', float(v)),
            "Maskalpha": lambda v: setattr(self, 'alpha', float(v))
        }
        
        # Call the appropriate handler if it exists
        if param_name in param_handlers:
            param_handlers[param_name](param_value)

    def par_exec_onPulse(self, par):
        param_name = par.name

        # Use the URL map directly to call the appropriate method or action
        action_map = {
            "Loadmodel": ext.TouchSamExt.load_model,
            "Reinit": lambda: None
        }

        # Call the action directly if it exists in the map
        action_map.get(param_name, lambda: self.about(param_name))()

    @staticmethod
    def about(endpoint):
        url_map = {
            "Github": "https://github.com/nicola-corbellini",
            "Instagram": "https://www.instagram.com/liminal_kin?igsh=MTZpcDRjOTAydGx0dw==",
            "Linkedin": "https://www.linkedin.com/in/nicola-corbellini-00780b1a9/"
        }     
        webbrowser.open(url_map[endpoint], new=2)

    @staticmethod
    def fifolog(status, message):
        current_time = datetime.now()
        formated_time = current_time.strftime("%H:%M:%S")
        op('fifo1').appendRow([formated_time, status, message])
