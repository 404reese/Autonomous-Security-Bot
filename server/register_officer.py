#!/usr/bin/env python3
"""
Register police officers for facial recognition.
Place officer photos in the 'officers/' folder and run this script.
"""

import os
import sys
from face_recognizer import OfficerFaceRecognizer

def register_officers_from_folder(folder_path: str = "officers"):
    """Register all officer photos from a folder"""
    recognizer = OfficerFaceRecognizer()
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"📁 Created '{folder_path}' folder. Please add officer photos there.")
        print("   Supported formats: .jpg, .jpeg, .png")
        print("   Filename format: 'Officer Name.jpg' (name will be extracted)")
        return
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    registered = 0
    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):
            # Extract officer name from filename (remove extension)
            name = os.path.splitext(filename)[0]
            # Replace underscores with spaces
            name = name.replace('_', ' ').replace('-', ' ')
            
            image_path = os.path.join(folder_path, filename)
            print(f"📸 Registering: {name} from {filename}...")
            
            try:
                recognizer.register_officer(image_path, name)
                registered += 1
            except Exception as e:
                print(f"   ❌ Failed: {e}")
    
    print(f"\n✅ Registered {registered} officers successfully!")
    
    if registered == 0:
        print("\n⚠️ No faces detected. Please ensure:")
        print("   - Photos contain clear, front-facing faces")
        print("   - Good lighting and resolution")

def register_single_photo(image_path: str, name: str):
    """Register a single officer photo"""
    recognizer = OfficerFaceRecognizer()
    recognizer.register_officer(image_path, name)
    print(f"✅ Registered {name} from {image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Register police officers for face recognition')
    parser.add_argument('--folder', type=str, default='officers', help='Folder containing officer photos')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--name', type=str, help='Officer name (for single image)')
    
    args = parser.parse_args()
    
    if args.image and args.name:
        register_single_photo(args.image, args.name)
    else:
        register_officers_from_folder(args.folder)