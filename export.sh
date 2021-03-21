#!/bin/sh
zip task.zip \
    queries.py \
    analysis.py \
    bitfinex_data.py \
    get_data.py \
    test_analysis.py \
    README.md \
    *.pickle \
    bitfinex_rates/*.csv \
    task.ipynb \
    task.html
