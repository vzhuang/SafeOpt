#!/bin/sh
python3 1safe_matern32.py 100 36 1 40 '1safe_32' 30 80 1 10
python3 1safe_matern52.py 100 36 1 40 '1safe_52' 30 80 1 10
python3 3safe_matern32.py 100 36 1 40 '3safe_32' 30 80 1 10
python3 3safe_matern52.py 100 36 1 40 '3safe_52' 30 80 1 10
python3 1safe_matern32_duel.py 100 36 1 40 '1safe_32_duel' 30 80 1 10
python3 1safe_matern52_duel.py 100 36 1 40 '1safe_52_duel' 30 80 1 10
