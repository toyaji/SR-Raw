#!/bin/sh

python data_maker.py -s 2 -p 100 --path "p96_scale2_np50"
python data_maker.py -s 3 -p 100 --path "p96_scale3_np50"
python data_maker.py -s 4 -p 100 --path "p96_scale4_np50"
python data_maker.py -s 8 -p 100 --path "p96_scale8_np50"
