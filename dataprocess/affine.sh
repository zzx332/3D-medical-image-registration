#! /bin/bash

#batch recon


function sum(){
	index=$((index+1))
}
function aff(){
    recon-all -i ./ixi/${file[$index]} -subjid ixi$index -autorecon1
}

index=0

##创建文件名数组
for i in `ls ./ixi/`;do
if [ $index -lt 11 ];then
file[$index]="$i" 
index=$((index+1))
fi
done
##按顺序运行十个程序
index=0
for x in `seq 1 10`;do
aff && echo ${file[$index]}+'/n'+'recon done!' && sum 
done 




