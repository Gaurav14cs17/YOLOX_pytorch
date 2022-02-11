import os
import shutil


path = "D:\labs\object_detection_model\old\Yolo-FastestV2\img"
output_path = ".\Riya"

if not os.path.exists(output_path):
    os.mkdir(output_path)

for x in os.listdir(path):
    full_path = os.path.join(path , x )
    print(full_path)
    #shutil.copy(full_path , output_path  )
    #shutil.move( s , d)
