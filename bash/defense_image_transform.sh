# CUDA_VISIBLE_DEVICES=0 python backdoor_defense.py --defense tabor --attack badnet --mark_alpha 0.0 --height 3 --width 3

work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
defense='image_transform'
transform_mode=$2

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${transform_mode}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size=3
alpha=0.0


attack='badnet'
echo "clean"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size --original --pretrain \
--transform_mode $transform_mode \
> $dirname/clean.txt 2>&1

for attack in 'badnet' 'latent_backdoor' 'trojannn' 'imc' 'reflection_backdoor' 'bypass_embed' 'clean_label' 'trojannet'
do
    echo $attack
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
    --dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size --pretrain \
    --transform_mode $transform_mode \
    > $dirname/${attack}.txt 2>&1
done

attack='badnet'
echo "targeted"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size \
--random_pos \
--transform_mode $transform_mode \
> $dirname/targeted.txt 2>&1