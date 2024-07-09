declare -a arr=( "aria_talebizadeh_eyeblink"  "aria_talebizadeh_smile" "aria_talebizadeh_fastalk" "aria_talebizadeh_mouthmove" "aria_talebizadeh_rotatemouth" \
		"arnefucks_eyeblink"  "arnefucks_smile" "arnefucks_fastalk" "arnefucks_mouthmove" "arnefucks_rotatemouth" \
		 "elias_wohlgemuth_eyeblink"  "elias_wohlgemuth_smile" "elias_wohlgemuth_fastalk" "elias_wohlgemuth_mouthmove" "elias_wohlgemuth_rotatemouth" \
		 "innocenzo_fulgintl_eyeblink"  "innocenzo_fulgintl_smile" "innocenzo_fulgintl_fastalk" "innocenzo_fulgintl_mouthmove" "innocenzo_fulgintl_rotatemouth" \
		 "mahabmarhai_eyeblink"  "mahabmarhai_smile" "mahabmarhai_fastalk" "mahabmarhai_mouthmove" "mahabmarhai_rotatemouth" \
		 "manuel_eyeblink" "manuel_smile" "manuel_fastalk" "manuel_mouthmove" "manuel_rotatemouth" \
	 	 "michaeldyer_eyeblink2"  "michaeldyer_smile2" "michaeldyer_fastalk2" "michaeldyer_mouthmove2" "michaeldyer_rotatemouth2" \
		  "seddik_houimli_eyeblink" "seddik_houimli_smile" "seddik_houimli_fastalk" "seddik_houimli_mouthmove" "seddik_houimli_rotatemouth" \
	)

for subj in "${arr[@]}"
do
    echo "$subj"
    # or do whatever with individual element of the array
    python lib/demo_seq.py experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py /cluster/balrog/jtang/Head_tracking/NPHM/dataset/recordings_process/$subj/color
done


