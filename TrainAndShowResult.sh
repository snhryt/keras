#!/bin/bash
set -e

PARENT_DIRPATH="/media/snhryt/Data/Research_Master/"
TARGET_DIRPATH="${PARENT_DIRPATH}keras/MyWork/capA_Augmentation/"
CLASS_NUM=100
BATCH_SIZE=256
EPOCHS=50

if [ ! -f "${TARGET_DIRPATH}model.hdf5" ]; then
  python python/TrainCaffenet.py \
         ${TARGET_DIRPATH} \
         ${CLASS_NUM} \
         --batch_size=${BATCH_SIZE} \
         --epochs=${EPOCHS} \
         -g
fi

FONT_LIST_FILEPATH="${PARENT_DIRPATH}Syn_AlphabetImages/selected/${CLASS_NUM}fonts/"
FONT_LIST_FILEPATH="${FONT_LIST_FILEPATH}SelectedFonts_${CLASS_NUM}class.txt"
ALPHABET="capA"

python python/ShowResult.py \
       ${TARGET_DIRPATH} \
       ${FONT_LIST_FILEPATH} \
       ${CLASS_NUM} \
       --alphabet=${ALPHABET}

# python python/DrawRecogResult.py /media/snhryt/Data/Research_Master/Real_Images/Characters_Resized_Selected/capA/ /media/snhryt/Data/Research_Master/keras/MyWork/capA_NoAugumentation/ -g