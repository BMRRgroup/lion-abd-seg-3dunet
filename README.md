# Abdominal Multi-organ Segmentation via 3DUNet
> Trained on LION data

## Running Instructions
1. Clone the repository
2. Download the prerequisites from environment.yml
3. From Releases
   - Download _3dunet_model_2_channel.pth.tar_ to '/TrainedModels/3DUNet_2_Channel'
4. Copy the inference dataset into '/Custom_Datasets/<Your_Dataset_Name\>'. This directory can have 2 filestructures

### Structure 1
Your_Dataset_Name
> Subjects_Dir
>> Subject_01
>>> Subject_01_Fat_fused.nii.gz </br>
>>> Subject_01_Water_fused.nii.gz </br>

>> ... </br>
>> Subject_ZZ
>>>...

### Structure 2
Your_Dataset_Name
> Subjects_Dir
>> Subject_01
>>> Subject_01_V1
>>>> Subject_01_V1_Fat_fused.nii.gz </br>
>>>> Subject_01_V1_Water_fused.nii.gz </br>

>>> Subject_01_V2
>>>> ...

>> ... </br>
>> Subject_ZZ
>>>...

5. Ensure that for each subject, the keywords denoting Fat, Water, T2* images remain identical.
6. While using the appropriate args, run _/Executables/predict_custom_input.py_
   - --dataset_name: Enter the name of the dataset <Your_Dataset_Name\>. By default, it is _Dataset_Name_
   - --fat_keyword: Enter the unique keyword for Fat maps. By default, it is _Fat_fused_
   - --water_keyword: Enter the unique keyword for Water maps. By default, it is _Water_fused_
   - --gpus GPU id. By default, it is set to 0
7. The images will be resized to (256, 224, 72) for the 3DUNet into '/Custom_Datasets/<Your_Dataset_Name\>/Interpolated_Subjects_Dir', and the corresponding segmentations into '/Custom_Datasets/<Your_Dataset_Name\>/Predicted_Masks'
