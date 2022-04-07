import os
import subprocess


def comp_video( videoName  , frameName ):
    output = subprocess.call('ffmpeg -i %s %s'% (videoName, frameName), shell=True)
    print(output)







dir_path = "/home/gaurav/Projects/temp_code/face_matrix/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/results/output_dir/"
output_dir = "/home/gaurav/Projects/temp_code/face_matrix/LFFD-A-Light-and-Fast-Face-Detector-for-Edge-Devices/results/cmp_video/"
for dir_name in os.listdir(dir_path):
    output_dir_path = os.path.join(output_dir , dir_name)
    os.makedirs(output_dir_path , exist_ok=True)
    for file_name in os.listdir(os.path.join(dir_path , dir_name)):
        full_path = os.path.join(dir_path , dir_name , file_name)
        output_dir_full_path = os.path.join(output_dir_path , file_name)
        comp_video(full_path , output_dir_full_path)
        print(full_path , output_dir_full_path)



