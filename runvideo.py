#!/usr/bin/env python3
"""
Multi-process parallel video processing launcher
Multi-process parallel video processing launcher – 8 CPU cores + GPU
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main function – launch the multi-process processing system"""
    
    print("=== Multi-process GPU Parallel Video Processing System ===")
    print("Fully utilizing 8 CPU cores + GPU for efficient video keypoint extraction")
    print()
    
    # Directly start the multi-process version
    print("Starting multi-process video processor...")
    try:
        from multiprocess_video_processor import main as mp_main
        mp_main()
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please make sure you are running in the correct directory and that dependencies are properly installed.")
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
