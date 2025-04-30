cfg_number = 7
path = "data"
mask="mask_removed" # can be "mask_removed" or "mask_removed_color" or "mask"


train_original_path = f"{path}/train/configuration1/"
train_modified_path = f"{path}/train/configuration{cfg_number}/"
test_original_path = f"{path}/test/configuration1/"
test_modified_path = f"{path}/test/configuration{cfg_number}/"
train_mask_path = f"{path}/{mask}/train/mask{cfg_number}/"
test_mask_path = f"{path}/{mask}/test/mask{cfg_number}/"