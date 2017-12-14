#!/usr/bin/env bash

python3 marioNetwork.py &

wait 5

ems=$(ps -ef | grep fceux)

while [ 1 ]; do
    if [ $(echo $ems | wc -l) -gt 1 ]; then
        id=$(echo $ems | head -n 1 | sed -E 's/[a-zA-z]+ +([0-9]+) +[0-9]/\1/g')
        kill $id
    fi
    ems=$(ps -ef | grep fceux)
done