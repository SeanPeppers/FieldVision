#!/bin/bash

: "

how to run maizaic:

./maizaic_run.sh -p /home/dek8v5/Documents/cornetv2/data_ori/FINAL_CORNETV2_DATASET/1_test/duplicate -h asift -d true

or 

./maizaic_run.sh --working_path /home/dek8v5/Documents/cornetv2/data_ori/FINAL_CORNETV2_DATASET/1_test/duplicate --hm_method asift --mode_duplicate true

"


#working_path="/data/e/cornetv2/maizaic_cs/24r_06_27_669_2passes_parallel"

#hm_method="cornetv3"

#angle_files=("$working_path"/*.csv)

#mode_duplicate=true

#declare

working_path=""
hm_method=""
mode_duplicate=true


while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--working_path) working_path="$2"; shift ;;
        -h|--hm_method) hm_method="$2"; shift ;;
        -d|--mode_duplicate) mode_duplicate="$2"; shift ;;  #use true or false
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done


if [ -z "$working_path" ] || [ -z "$hm_method" ]; then
    echo "Usage: ./maizaic_run.sh -w <working_path> -h <hm_method> [-d <mode_duplicate>]"
    exit 1
fi


angle_files=("$working_path"/*.csv)

#calibrate
if [ "$mode_duplicate" = "true" ]; then
    save_path="$working_path/calibrated"
elif [ "$mode_duplicate" = "false" ]; then
    save_path="$working_path/raw"
else
    echo "Invalid value for mode_duplicate. Use 'true' or 'false'."
    exit 1
fi
:'
python calibration.py \
     -image_path  "$working_path/raw" \
     -save_path "$save_path"
'
start_time=$(date +%s)

#split to group based on boundaries

split="python split_for_mini.py \
     -image_path \"$save_path\" \
     -save_path \"$working_path/${hm_method}_mini_partition\" \
     -hm \"$working_path/homography_matrices/H_${hm_method}.csv\" \
     -angle_csv ${angle_files[@]}"

if [ "$mode_duplicate" = "true" ]; then
    split="$split -duplicate"
elif [ "$mode_duplicate" = "false" ]; then
    echo "Running script without the -duplicate flag."
else
    echo "Warning: Unrecognized value for mode_duplicate. Expected 'true' or 'false'."
fi

eval $split

#loop to mosaic all all minis

mini_path="$working_path/${hm_method}_mini_partition"

for mini_hm in "$mini_path"/*.csv;
do
    if [ -f "$mini_hm" ]; then
        echo "processing $mini_hm"

        #extract the group number from the file name
        group_number=$(basename "$mini_hm" | grep -oE '[0-9]+')

        python stitcher.py \
            -image_path "$mini_path/group_$group_number" \
            -hm "$mini_hm" \
            -save_path "$working_path/${hm_method}_mini_mosaics" \
            -scale 3 \
            -fname "group$group_number" \
            -mini_mosaic
    else
        echo "there is no .csv file found in $mini_hm"
    fi
done



#assemble global mosaic
#surf_assembly_global.py \

python mini_mosaic_360.py \
       -image_path "$working_path/${hm_method}_mini_mosaics" \
       -save_path "$working_path/${hm_method}_global_mosaic"


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Elapsed time: $elapsed seconds"
