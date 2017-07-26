#!/bin/bash
set -e

PARENT_DIRPATH="/media/snhryt/Data/Research_Master"

# --Character classification--------------------------------
TARGET_DIRPATH="${PARENT_DIRPATH}/keras/MyWork/CharacterClassification/Syn-Real_Augmentation_Lenet"
CLASS_NUM=26
BATCH_SIZE=64
EPOCHS=50
if [ ! -f "${TARGET_DIRPATH}/model.hdf5" ]; then
  python python/train.py \
         ${TARGET_DIRPATH} \
         ${CLASS_NUM} \
         --batch_size=${BATCH_SIZE} \
         --epochs=${EPOCHS} \
         -g \
         -a
fi

TEST_IMG_DIRPATH="${PARENT_DIRPATH}/Real_Images/Characters_Resized_Selected/Arts_Photography"
python python/DrawRecogResult.py \
       ${TEST_IMG_DIRPATH} \
       ${TARGET_DIRPATH}
# ---------------------------------------------------------

# --Font shape classification------------------------------
# TARGET_DIRPATH="${PARENT_DIRPATH}/keras/MyWork/FontShapeClassification/Syn-Real_NoAugmentation_Caffenet"
# CLASS_NUM=100
# BATCH_SIZE=256
# EPOCHS=50
# if [ ! -f "${TARGET_DIRPATH}/model.hdf5" ]; then
#   python python/train.py \
#          ${TARGET_DIRPATH} \
#          ${CLASS_NUM} \
#          --batch_size=${BATCH_SIZE} \
#          --epochs=${EPOCHS} \
#          -g
# fi

# TEST_IMG_DIRPATH="${PARENT_DIRPATH}/Real_Images/Characters_Resized_Selected/Arts & Photography"
# python python/DrawRecogResult.py \
#        ${TEST_IMG_DIRPATH} \
#        ${TARGET_DIRPATH}
# ---------------------------------------------------------


