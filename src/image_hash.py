import base64
import hashlib, os
import imagehash
from PIL import Image
import io


class ImageHash:
    def __init__(self):
        self.cnt=0
        self.previous_hash=""
        print("Image is parsed for hash comparison")
    
    def get_next_filename(self, directory: str, base_name: str, extension: str) -> str:
        index = 1
        while True:
            filename = f"{base_name}{index}{extension}"
            filepath = os.path.join(directory, filename)
            if not os.path.exists(filepath):
                return filename
            index += 1
   
    def imageOutput(self, img) -> bool :
        
       
        try:
            image = Image.open(io.BytesIO(img))
            
            image_hash = imagehash.average_hash(image)

            # print("hash value", image_hash)

            if self.cnt > 0:
                
                previous_key = self.previous_hash
                current_key = image_hash
                if current_key == previous_key:
                        return False
            self.cnt+=1
            self.previous_hash = image_hash
    
        except FileNotFoundError:
            print("Image file not found. Please check the path.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        return True

