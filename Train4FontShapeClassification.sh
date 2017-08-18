#!/bin/bash
set -e

PARENT_DIRPATH="/media/snhryt/Data/Research_Master"
SELECTED_FONT_DIRPATH="${PARENT_DIRPATH}/Syn_AlphabetImages/selected"

CLASS_NUM=50
NETWORK="lenet"
ALPHABET="all"
SELECTION_METHOD="${CLASS_NUM}fonts_ClusteringRegular"
TARGET_DIRNAME="${ALPHABET}_Augmentation_${NETWORK}"

TARGET_DIRPATH="${PARENT_DIRPATH}/keras/MyWork/FontShapeClassification/${SELECTION_METHOD}"
TARGET_DIRPATH="${TARGET_DIRPATH}/${TARGET_DIRNAME}"
if [ ! -f "${TARGET_DIRPATH}/train.txt" ]; then
  MyWork/FontShapeClassification/MakeFilepathsList \
    "${SELECTED_FONT_DIRPATH}/${SELECTION_METHOD}/SelectedFonts.txt" \
    ${TARGET_DIRPATH} \
    ${ALPHABET} \
    "false"
fi

BATCH_SIZE=256
EPOCHS=50
if [ ! -f "${TARGET_DIRPATH}/model.hdf5" ]; then
  python python/train.py \
         ${TARGET_DIRPATH} \
         ${CLASS_NUM} \
         --batch_size=${BATCH_SIZE} \
         --epochs=${EPOCHS} \
         --network=${NETWORK} \
         -g \
         -a
fi

# TEST_PATH="${PARENT_DIRPATH}/Real_Images/Characters_Resized_Selected/Arts_Photography"
TEST_PATH="${PARENT_DIRPATH}/Real_Images/Characters_Resized_Selected/capA"
FONT_IMG_DIRPATH="${PARENT_DIRPATH}/Syn_AlphabetImages/font"
FONT_LIST_FILEPATH="${PARENT_DIRPATH}/Syn_AlphabetImages/selected/${SELECTION_METHOD}"
FONT_LIST_FILEPATH="${FONT_LIST_FILEPATH}/SelectedFonts.txt"
python python/DrawRecogResult.py \
       ${TEST_PATH} \
       ${TARGET_DIRPATH} \
       --classification_target="font" \
       --medoid_fonts_list_filepath=${FONT_LIST_FILEPATH} \
       --char_imgs_dirpath=${FONT_IMG_DIRPATH}

       