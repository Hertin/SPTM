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
# FRNAME=ph_gt
# FRNAME=ph_mfa
# FRNAME=ph_g2pw
FRNAME=ph_g2pu
DATASET=p2u/aishell3
LABEL=${FRNAME}
MANIFSET=manifest/${DATASET}
TEXT=${MANIFSET}
DICT=$TEXT/dict.${LABEL}.txt

# FRNAME=ph_g2puv1
# FRNAME=ph_g2puv2
# FRNAME=ph_g2puv3
# FRNAME=ph_g2puv4
# FRNAME=ph_g2puv5
# FRNAME=ph_g2puv6
# FRNAME=ph_gt
# FRNAME=ph_mfa
# FRNAME=ph_kakasi
# FRNAME=ph_randlex
# FRNAME=ph_g2pu
# DATASET=p2u/csj
# LABEL=${FRNAME}
# MANIFSET=manifest/${DATASET}
# TEXT=${MANIFSET}
# DICT=$TEXT/dict.${LABEL}.txt

# LABEL=ph_g2puv1_ipa
# TEXT=manifest/p2u/aishell3
# DICT=manifest/p2u/dict.ipa.txt

# LABEL=ipa
# TEXT=manifest/librispeech100
# DICT=manifest/p2u/dict.ipa.txt

fairseq-preprocess \
    --only-source \
    --srcdict ${DICT} \
    --trainpref ${TEXT}/train.${LABEL} \
    --validpref ${TEXT}/dev.${LABEL} \
    --testpref ${TEXT}/test.${LABEL}  \
	--destdir data-bin/$(basename ${TEXT})_${LABEL} \
    --workers 32

