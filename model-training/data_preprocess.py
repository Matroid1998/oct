import os
import shutil
import pandas as pd
import numpy as np
import random

def zh_dataset(dataset_path, output_folder,dataset_name):
    os.makedirs(output_folder, exist_ok=True)
    for foldername in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, foldername)):
            print(foldername)
            for file_name in os.listdir(os.path.join(dataset_path,foldername)):
                parts = file_name.split('-')
                if len(parts) != 3:
                    print(file_name)
                disease = parts[0]
                patient_id = parts[1]
                image_number = parts[2]
                new_name = f"{dataset_name}_{disease}_{patient_id}_{image_number}"
                old_path = os.path.join(dataset_path,foldername, file_name)
                new_path = os.path.join(output_folder, new_name)
                shutil.copy(old_path, new_path)
def NEH_dataset(dataset_path, output_folder , dataset_name):
    df = pd.read_csv('data_information.csv')
    df['Directory'] = df['Directory'].str.replace('/','\\')
    os.makedirs(output_folder,exist_ok = True)
    num=0
    random_numbers = set()
    for disease in os.listdir(dataset_path):
        print(disease)
        for patient_id in os.listdir(os.path.join(dataset_path,disease)):
            new_patient_id = generate_unique_patient_id(random_numbers)
            random_numbers.add(new_patient_id)
            image_number = 0
            if 'OS' and 'OD' in os.listdir(os.path.join(dataset_path,disease,patient_id)):
                for image_name in os.listdir(os.path.join(dataset_path,disease,patient_id,'OS')):
                    extension = image_name.split('.')[1]
                    formatted_number = f"{image_number:03d}"
                    new_name = f'{dataset_name}_{disease}_{new_patient_id}_{formatted_number}.{extension}'
                    old_path = os.path.join(os.path.join(dataset_path,disease,patient_id,'OS',image_name))
                    new_path = os.path.join(output_folder,new_name)
                    result_row = df[df['Directory'].str.strip() == os.path.join(disease,patient_id,'OS',image_name).strip()]
                    if result_row['Label'].values[0] == result_row['Class'].values[0]:
                        shutil.copy(old_path,new_path)
                        image_number += 1
                        num += 1
                for image_name in os.listdir(os.path.join(dataset_path,disease,patient_id,'OD')):
                    extension = image_name.split('.')[1]
                    formatted_number = f"{image_number:03d}"
                    new_name = f'{dataset_name}_{disease}_{new_patient_id}_{formatted_number}.{extension}'
                    old_path = os.path.join(os.path.join(dataset_path,disease,patient_id,'OD',image_name))
                    new_path = os.path.join(output_folder,new_name)
                    result_row = df[df['Directory'].str.strip() == os.path.join(disease,patient_id,'OD',image_name).strip()]
                    if result_row['Label'].values[0] == result_row['Class'].values[0]:
                        shutil.copy(old_path,new_path)
                        image_number += 1
                        num += 1
            else:
                for image_name in os.listdir(os.path.join(dataset_path,disease,patient_id)):
                    image_number,di = image_name.split('_')
                    extension = di.split('.')[1]
                    new_name = f'{dataset_name}_{disease}_{new_patient_id}_{image_number}.{extension}'
                    old_path = os.path.join(dataset_path,disease,patient_id,image_name)
                    new_path = os.path.join(output_folder,new_name)
                    result_row = df[df['Directory'].str.strip() == os.path.join(disease,patient_id,image_name).strip()]
                    if result_row['Label'].values[0] == result_row['Class'].values[0]:
                        shutil.copy(old_path,new_path)
                        num += 1
    return num
def generate_unique_patient_id(existing_ids):
    new_id = str(random.randint(1000, 9999))
    while new_id in existing_ids:
        new_id = str(random.randint(1000, 9999))
    return new_id

if __name__ == "__main__":
    dataset_name = 'zhang'
    dataset_path = 'D:\\ZhangLabData\\CellData\\OCT\\train'
    output_folder = 'D:\oct_all_data'
    zh_dataset(dataset_path, output_folder,dataset_name)
    # dataset_name = 'NEH'
    # dataset_path = 'D:\\NEH_UT_2021RetinalOCTDataset\\NEH_UT_2021RetinalOCTDataset'
    # output_folder = 'D:\\oct_all_data'
    # nhi = NEH_dataset(dataset_path, output_folder,dataset_name)
    # print(nhi)