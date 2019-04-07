#!/bin/bash

for ((i = 0; i <= $1; ++i))
do
    p=$(printf "%05d" $i)
    convert "dump_"$i".ppm" -quality 0 "it_${p}.png"
done

convert "it_"*".png" -delay 300 movie.gif
#convert movie.gif -resize 2000x2000 movie.gif
