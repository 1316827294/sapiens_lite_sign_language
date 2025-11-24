#!/usr/bin/env python3
"""
Multi-process GPU parallel video keypoint extraction system

Efficient processing scheme using 8 CPU cores for video decoding
and preprocessing + GPU batch inference.
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
import time
from tqdm import tqdm
import warnings
from typing import List, Tuple, Dict, Any
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import psutil

# Import required modules from existing codebase
from demo.classes_and_palettes import (
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO
)
from demo.pose_utils import nms, top_down_affine_transform, udp_decode

try:
    from mmdet.apis import inference_detector, init_detector
    from detector_utils import (
        adapt_mmdet_pipeline,
        init_detector,
        process_images_detector,
    )
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

warnings.filterwarnings("ignore")

@dataclass
class FrameBatch:
    """Frame batch data structure"""
    video_path: str
    frames: List[np.ndarray]
    frame_indices: List[int]
    batch_id: int

@dataclass
class ProcessingResult:
    """Processing result data structure"""
    video_path: str
    frame_idx: int
    timestamp: float
    instances: List[Dict]

class GPUInferenceWorker:
    """GPU inference worker – singleton-like behavior to ensure a single GPU thread"""
    
    def __init__(self, pose_checkpoint: str, device: str = "cuda:0", batch_size: int = 64):
        self.pose_checkpoint = pose_checkpoint
        self.device = device
        self.batch_size = batch_size
        self.input_shape = (3, 1024, 768)
        self.heatmap_scale = 4
        
        # Initialize model
        self._init_model()
        
        # Setup queues and thread – reduce queue size to save memory
        self.input_queue = queue.Queue(maxsize=20)
        self.output_queue = queue.Queue(maxsize=20)
        self.running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()
        
    def _init_model(self):
        """Initialize GPU model"""
        print(f"Initializing GPU model on {self.device}...")
        
        USE_TORCHSCRIPT = '_torchscript' in self.pose_checkpoint or self.pose_checkpoint.endswith('.pt2')
        
        if USE_TORCHSCRIPT:
            print("Loading TorchScript model...")
            self.pose_estimator = torch.jit.load(self.pose_checkpoint, map_location=self.device)
            self.dtype = torch.float32
            print("✓ TorchScript model loaded")
        else:
            print("Loading PyTorch model...")
            self.pose_estimator = torch.export.load(self.pose_checkpoint).module()
            self.pose_estimator = self.pose_estimator.to(self.device)
            self.dtype = torch.bfloat16
            print("✓ PyTorch model loaded")
        
        self.pose_estimator.eval()
        
        # GPU warmup
        self._warmup_gpu()
        print("✓ GPU model initialized")
    
    def _warmup_gpu(self):
        """GPU warmup – simplified version"""
        print("Warming up GPU...")
        try:
            if not torch.cuda.is_available():
                print("CUDA not available, skipping warmup")
                return
                
            print("Creating small warmup batch...")
            dummy_input = torch.randn(
                2, 3, self.input_shape[1], self.input_shape[2], 
                device=self.device, dtype=self.dtype
            )
            
            print("Running warmup inference...")
            with torch.no_grad():
                _ = self.pose_estimator(dummy_input)
            
            torch.cuda.empty_cache()
            print("✓ GPU warmup completed")
        except Exception as e:
            print(f"GPU warmup failed, continuing anyway: {e}")
    
    def _preprocess_frames_gpu(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Preprocess multiple frames on GPU"""
        preprocessed = []
        
        mean = torch.tensor([123.5, 116.5, 103.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([58.5, 57.0, 57.5], device=self.device).view(1, 3, 1, 1)
        
        for frame in frames:
            img, center, scale = top_down_affine_transform(
                frame.copy(), 
                np.array([0, 0, frame.shape[1], frame.shape[0]])
            )
            
            img = cv2.resize(
                img, (self.input_shape[2], self.input_shape[1]),
                interpolation=cv2.INTER_LINEAR
            ).transpose(2, 0, 1)
            
            img_tensor = torch.from_numpy(img).float().to(self.device)
            img_tensor = img_tensor[[2, 1, 0], ...]  # BGR → RGB
            
            preprocessed.append(img_tensor)
        
        if len(preprocessed) > 0:
            batch_tensor = torch.stack(preprocessed, dim=0)
            batch_tensor = (batch_tensor - mean) / std
            return batch_tensor
        
        return torch.empty(0, 3, self.input_shape[1], self.input_shape[2], device=self.device)
    
    def _inference_loop(self):
        """Main GPU inference loop"""
        while self.running:
            try:
                batch_data = self.input_queue.get(timeout=1.0)
                if batch_data is None:
                    break
                
                frames_batch, video_info_batch = batch_data
                
                all_frames = []
                batch_info = []
                
                for video_path, frames, frame_indices in zip(*video_info_batch):
                    all_frames.extend(frames)
                    for i, frame_idx in enumerate(frame_indices):
                        batch_info.append((video_path, frame_idx, i))
                
                if len(all_frames) == 0:
                    continue
                
                processed_tensor = self._preprocess_frames_gpu(all_frames)
                
                if processed_tensor.size(0) == 0:
                    continue
                
                # Batch GPU inference to avoid OOM
                all_heatmaps = []
                for i in range(0, processed_tensor.size(0), self.batch_size):
                    batch_tensor = processed_tensor[i:i+self.batch_size]
                    
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                        heatmaps = self.pose_estimator(batch_tensor)
                        all_heatmaps.append(heatmaps)
                
                if all_heatmaps:
                    combined_heatmaps = torch.cat(all_heatmaps, dim=0)
                    
                    results = []
                    for i, (video_path, frame_idx, _) in enumerate(batch_info):
                        if i < combined_heatmaps.size(0):
                            heatmap = combined_heatmaps[i].cpu()
                            
                            result = udp_decode(
                                heatmap.unsqueeze(0).float().data[0].numpy(),
                                (self.input_shape[1], self.input_shape[2]),
                                (int(self.input_shape[1] / self.heatmap_scale), 
                                 int(self.input_shape[2] / self.heatmap_scale))
                            )
                            
                            keypoints, keypoint_scores = result
                            
                            frame_shape = all_frames[i].shape
                            scale_x = frame_shape[1] / self.input_shape[2]
                            scale_y = frame_shape[0] / self.input_shape[1]
                            
                            if len(keypoints.shape) == 3:
                                keypoints = keypoints[0]
                            
                            keypoints[:, 0] *= scale_x
                            keypoints[:, 1] *= scale_y
                            
                            if hasattr(keypoint_scores, 'tolist'):
                                scores_list = keypoint_scores.tolist()
                            else:
                                scores_list = keypoint_scores
                            
                            if isinstance(scores_list, list) and len(scores_list) > 0 and isinstance(scores_list[0], list):
                                flat_scores = []
                                for score_item in scores_list:
                                    if isinstance(score_item, list):
                                        flat_scores.extend(score_item)
                                    else:
                                        flat_scores.append(score_item)
                                scores_list = flat_scores
                            
                            result_data = {
                                "keypoints": keypoints.tolist(),
                                "keypoint_scores": scores_list,
                            }
                            
                            results.append((video_path, frame_idx, result_data))
                    
                    self.output_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"GPU inference error: {e}")
                continue
    
    def add_batch(self, frames_batch: List[np.ndarray], video_info_batch: Tuple):
        """Add batch to inference queue"""
        self.input_queue.put((frames_batch, video_info_batch))
    
    def get_results(self):
        """Fetch inference results"""
        try:
            return self.output_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop inference worker"""
        self.running = False
        self.input_queue.put(None)
        self.inference_thread.join()

def video_frame_loader(video_path: str, frame_skip: int = 1, max_frames_per_batch: int = 32):
    """Video frame loader – CPU-intensive task"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_batch = []
    frame_indices_batch = []
    frame_idx = 0
    
    video_info = {
        "path": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "frame_skip": frame_skip
    }
    
    batches = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip == 0:
            frames_batch.append(frame.copy())
            frame_indices_batch.append(frame_idx)
            
            if len(frames_batch) >= max_frames_per_batch:
                batches.append({
                    'video_info': video_info,
                    'frames': frames_batch.copy(),
                    'frame_indices': frame_indices_batch.copy()
                })
                frames_batch.clear()
                frame_indices_batch.clear()
        
        frame_idx += 1
    
    if frames_batch:
        batches.append({
            'video_info': video_info,
            'frames': frames_batch,
            'frame_indices': frame_indices_batch
        })
    
    cap.release()
    return batches

class MultiProcessVideoProcessor:
    """Multi-process video processor – utilizes 8 CPU cores + GPU"""
    
    def __init__(self, 
                 pose_checkpoint: str,
                 device: str = "cuda:0",
                 num_workers: int = 8,
                 gpu_batch_size: int = 128,
                 frame_batch_size: int = 64):
        
        self.pose_checkpoint = pose_checkpoint
        self.device = device
        self.num_workers = min(num_workers, psutil.cpu_count())
        self.gpu_batch_size = gpu_batch_size
        self.frame_batch_size = frame_batch_size
        
        print(f"Using {self.num_workers} CPU workers and {device} for GPU inference")
        
        self.gpu_worker = GPUInferenceWorker(
            pose_checkpoint=pose_checkpoint,
            device=device,
            batch_size=gpu_batch_size
        )
        
        self.results_queue = queue.Queue()
        self.processed_videos = {}
        
    def process_video(self, video_path: str, output_path: str, frame_skip: int = 1):
        """Process a single video"""
        print(f"Processing: {video_path}")
        
        batches = video_frame_loader(video_path, frame_skip, self.frame_batch_size)
        if not batches:
            print(f"Failed to load video: {video_path}")
            return False
        
        video_info = batches[0]['video_info']
        video_results = {
            "video_info": video_info,
            "frames": {}
        }
        
        total_frames_to_process = sum(len(batch['frames']) for batch in batches)
        processed_count = 0
        
        with tqdm(total=total_frames_to_process, desc=f"Processing {Path(video_path).name}") as pbar:
            
            for batch in batches:
                frames = batch['frames']
                frame_indices = batch['frame_indices']
                
                self.gpu_worker.add_batch(
                    frames,
                    ([video_path], [frames], [frame_indices])
                )
            
            collected_batches = 0
            while collected_batches < len(batches):
                results = self.gpu_worker.get_results()
                if results:
                    for video_path_result, frame_idx, keypoints_data in results:
                        if keypoints_data:
                            video_results["frames"][frame_idx] = {
                                "timestamp": frame_idx / video_info['fps'],
                                "instances": [keypoints_data]
                            }
                        processed_count += 1
                        pbar.update(1)
                    collected_batches += 1
                else:
                    time.sleep(0.01)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(video_results, f, indent=2)
        
        print(f"✓ Processed {processed_count} frames, saved to: {output_path}")
        return True
    
    def process_dataset(self, dataset_root: str, output_root: str, splits: List[str], frame_skip: int = 1):
        """Process the entire dataset"""
        print(f"Processing dataset with {self.num_workers} workers...")
        
        all_videos = []
        for split in splits:
            split_path = Path(dataset_root) / split
            if split_path.exists():
                for category_dir in split_path.iterdir():
                    if category_dir.is_dir():
                        for video_file in category_dir.glob("*.mp4"):
                            relative_path = video_file.relative_to(dataset_root)
                            output_path = Path(output_root) / relative_path.with_suffix('.json')
                            
                            if not output_path.exists():
                                all_videos.append((str(video_file), str(output_path)))
        
        print(f"Found {len(all_videos)} videos to process")
        
        processed_count = 0
        failed_count = 0
        
        max_concurrent_videos = 2
        
        with tqdm(total=len(all_videos), desc="Processing videos") as pbar:
            with ThreadPoolExecutor(max_workers=max_concurrent_videos) as executor:
                
                batch_size = 100
                video_idx = 0
                
                while video_idx < len(all_videos):
                    current_batch = all_videos[video_idx:video_idx + batch_size]
                    
                    future_to_video = {
                        executor.submit(self.process_video, video_path, output_path, frame_skip): (video_path, output_path)
                        for video_path, output_path in current_batch
                    }
                    
                    for future in as_completed(future_to_video):
                        video_path, output_path = future_to_video[future]
                        try:
                            success = future.result(timeout=3600)
                            if success:
                                processed_count += 1
                                print(f"✓ Successfully processed: {Path(video_path).name}")
                            else:
                                failed_count += 1
                                print(f"✗ Failed to process: {video_path}")
                        except Exception as e:
                            failed_count += 1
                            print(f"✗ Error processing {video_path}: {e}")
                            import traceback
                            traceback.print_exc()
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Processed': processed_count, 
                            'Failed': failed_count,
                            'Remaining': len(all_videos) - processed_count - failed_count
                        })
                    
                    video_idx += batch_size
                    
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {processed_count}")
        print(f"Failed: {failed_count}")
        
        self.gpu_worker.stop()
        
        return processed_count, failed_count

def main():
    """Main function – multi-process parallel video processing"""
    
    # Configuration (modify for your environment)
    POSE_CHECKPOINT = "/home/yunli/data/sapiens/lite/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
    DATASET_ROOT = "/shares/iict-sp2.ebling.cl.uzh/common/popsign_v1_0/game/"
    OUTPUT_ROOT = "/home/yunli/data/sapiens/lite/game/"
    
    # Performance tuning – balance between speed and memory usage
    NUM_WORKERS = 4
    GPU_BATCH_SIZE = 32
    FRAME_BATCH_SIZE = 16
    FRAME_SKIP = 1
    SPLITS = ["train", "val", "test"]
    
    print("=== Multi-process GPU Parallel Video Processing System ===")
    print(f"CPU workers: {NUM_WORKERS}")
    print(f"GPU batch size: {GPU_BATCH_SIZE}")
    print(f"Frame batch size: {FRAME_BATCH_SIZE}")
    print(f"Frame skip: {FRAME_SKIP}")
    print()
    
    mp.set_start_method('spawn', force=True)
    
    processor = MultiProcessVideoProcessor(
        pose_checkpoint=POSE_CHECKPOINT,
        device="cuda:0",
        num_workers=NUM_WORKERS,
        gpu_batch_size=GPU_BATCH_SIZE,
        frame_batch_size=FRAME_BATCH_SIZE
    )
    
    start_time = time.time()
    processed, failed = processor.process_dataset(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        splits=SPLITS,
        frame_skip=FRAME_SKIP
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per video: {total_time / max(processed, 1):.2f} seconds")

if __name__ == "__main__":
    main()
