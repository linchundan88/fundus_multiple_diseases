#!/bin/bash

#source activate dlp_new
#sleep 6s

#read -s -n1 -p "press any key to continue ... "

cd ~/dlp/PycharmProjects/fundus_multiple_diseases/RPC

#python RPC_server_single_class.py -5 19995 &  #img_position
python RPC_server_single_class.py -4 19996 &  #left right eye
python RPC_server_single_class.py -3 19997 &  #gradable
python RPC_server_single_class.py -1 19999 &  #fundus, ocular surface, others
python RPC_server_single_class.py 0 20000 &  #BigClass
python RPC_server_single_class.py 0_1 20001 &
python RPC_server_single_class.py 0_2 20002 &
python RPC_server_single_class.py 0_3 20003 &
python RPC_server_single_class.py 1 20010 &  #DR2_3
python RPC_server_single_class.py 2 20020 &  #RVO
python RPC_server_single_class.py 5 20050 &  #CSCR, VKH disease
python RPC_server_single_class.py 10 20100 &  #Possible glaucoma, Optic atrophy
python RPC_server_single_class.py 15 20150 &  # Retinitis pigmentosa, Bietti crystalline dystrophy
python RPC_server_single_class.py 29 20290 &  #BLUR

python RPC_server_single_class.py 60 20600 &  #Neovascularization

python RPC_server_optic_disc.py 21000 &

#python RPC_server_lesions_seg.py 22000 &

python RPC_server_CAM.py 0 23000 &   #bigclass, multi_class,port
python RPC_server_CAM.py 1 23001 &   #bigclass, multi_label,port

python RPC_server_deep_explain.py 0 0 24000 &  #bigclass, model_no, port
python RPC_server_deep_explain.py 0 1 24001 &

#python RPC_server_deep_shap.py 0_10 25000 &    #bigclass multi_class

