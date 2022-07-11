#!/bin/sh

python data_maker.py -s 2 -p 100 --path "p96_x2"
python data_maker.py -s 3 -p 100 --path "p96_x3"
python data_maker.py -s 4 -p 100 --path "p96_x4"
python data_maker.py -s 8 -p 100 --path "p96_x8"
