#!/bin/bash

# FRNAME=phg2pw
# REPEAT=15
# # REPEAT=30
# # REPEAT=75
# DATASET=p2u/aishellX
# LABEL=${FRNAME}${REPEAT}
# MANIFSET=manifest/${DATASET}
# TEXT=${MANIFSET}/${LABEL}
# DICT=$TEXT/dict.txt.txt
# fairseq-preprocess \
#     --only-source \
#     --srcdict ${DICT} \
#     --trainpref ${TEXT}/train.txt \
#     --validpref ${TEXT}/valid.txt \
#     --testpref ${TEXT}/test.txt \
# 	--destdir data-bin/$(basename ${DATASET})_${LABEL} \
#     --workers 32


# FRNAME=ph_g2puv1
# FRNAME=ph_g2puv2
# FRNAME=ph_g2puv3
# FRNAME=ph_g2puv4
# FRNAME=ph_g2puv5
# FRNAME=ph_g2puv6
# REPEAT=15
# DATASET=p2u/aishell3
# LABEL=${FRNAME}_${REPEAT}
# MANIFSET=manifest/${DATASET}
# TEXT=${MANIFSET}/${LABEL}
# DICT=$TEXT/dict.${LABEL}.txt

FRNAME=ipa
REPEAT=15
DATASET=librispeech100
LABEL=${FRNAME}${REPEAT}
MANIFSET=manifest/${DATASET}
TEXT=${MANIFSET}/${LABEL}
# DICT=$TEXT/dict.${LABEL}.txt
DICT=manifest/p2u/dict.${LABEL}.txt

fairseq-preprocess \
    --only-source \
    --srcdict ${DICT} \
    --trainpref ${TEXT}/train.${LABEL} \
    --validpref ${TEXT}/dev.${LABEL} \
    --testpref ${TEXT}/test.${LABEL}  \
	--destdir data-bin/$(basename ${DATASET})_${LABEL} \
    --workers 32

