#!/bin/bash
set -e

PARENT_DIRPATH="/media/snhryt/Data/Research_Master"

TARGET_DIRPATH="${PARENT_DIRPATH}/keras/MyWork/CharacterClassification/Syn-Real_NoAugmentation_Caffenet"
CLASS_NUM=26
BATCH_SIZE=256
EPOCHS=50
if [ ! -f "${TARGET_DIRPATH}/model.hdf5" ]; then
  python python/train.py \
         ${TARGET_DIRPATH} \
         ${CLASS_NUM} \
         --batch_size=${BATCH_SIZE} \
         --epochs=${EPOCHS} \
         --network="caffenet" \
         -g
fi

TEST_PATH="${PARENT_DIRPATH}/keras/MyWork/CharacterClassification/TestImages.txt"
python python/DrawRecogResult.py \
       ${TEST_PATH} \
       ${TARGET_DIRPATH} \
       --classification_target="character"



