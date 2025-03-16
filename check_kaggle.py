import kaggle
kaggle.api.authenticate()
kaggle.api.dataset_download_files('yasserh/kinematics-motion-data', path='./data', unzip=True)


