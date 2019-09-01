#! /bin/bash


filename="Task08_HepaticVessel.tar"

file_id="1qVrpV7vmhIsUxFiH189LmAn0ALbAPrgS"

query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}"| perl -nE'say/uc-download-link.*? href="(.*?)\">/'| sed -e 's/amp;//g' | sed -n 2p`

url="https://drive.google.com$query"

curl -b ./cookie.txt -L -o ${filename} $url
