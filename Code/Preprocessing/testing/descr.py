import Code.Libraries.MyOculusLib as mol

path = '../../../Images/data/'
mol.init(im_path=path, mouse_f=mol.draw_elipse)

data_process_keys= {"ignore_keys": "age.txt",
                    "merge_keys": "description1.txt",
                    "split_to_words_keys": "description1.txt",
                    "n_gram_keys": "description1.txt",
                    "merge_mapping_keys": "description1.txt",
                    "merge_mapping_keys_ratio": 0.65,
                    "merge_mapping_keys_save": True,
                    "merge_mapping_keys_load": True,
                    "merge_mapping_keys_path": "merge_mapping.csv",
                    "merge_mapping_keys_accepted_path": "accepted_mapping.csv",
                    "merge_mapping_keys_rejected_path": "rejected_mapping.csv",
                    "save_total_keys": "description1.txt"}
mol.all_path(mol.get_description_full_path, eye='left', data_once_per_patient_eye_tuple=(True, True), collect_data=True, data_process_keys=data_process_keys)
