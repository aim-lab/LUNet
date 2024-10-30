from PVBM.Datasets import PVBMDataDownloader
path_to_save_datasets = "Databases"
dataset_downloader = PVBMDataDownloader()
dataset_downloader.download_dataset(name="Crop_HRF", save_folder_path=path_to_save_datasets)
dataset_downloader.download_dataset(name="INSPIRE", save_folder_path=path_to_save_datasets)
#dataset_downloader.download_dataset(name="UNAF", save_folder_path=path_to_save_datasets) #LUNet paper was evaluated
#on the crop version of UNAF that haven't been published. You can still evaluate it on the original UNAF dataset but
#results may be different.
print("Images downloaded successfully")