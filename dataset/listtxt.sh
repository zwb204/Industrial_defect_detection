#!/bin/bash

for subdir in `ls`
do
    echo $subdir
    if [ -d "$subdir" ]
        then
            for file in `ls $subdir`
            do
                echo "$subdir/$file" >> list.txt
            done
         else
             echo "pass"
    fi
done
