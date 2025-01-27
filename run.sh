cd data

ds="Magazine"
dn="Magazine_5" 
rr=0.7

for mode in "MISS" #"ORIG" 
do 
    # python process_N.py --dataset $ds  --data_name $dn --mode MISS --review_rate $rr

    cd ../


    CUDA_VISIBLE_DEVICES=7 python Edgeformer-N/main.py --data_path "data/${ds}/${ds}_rr_${rr}/${mode}"

done


# for mode in "LLM"
# do 
#     python process_N.py --dataset $ds  --data_name $dn --mode MISS
#     cd ../


#     CUDA_VISIBLE_DEVICES=0 python Edgeformer-N/main.py --data_path data/Appliances/Appliances_rr_0.3/miss

# done


