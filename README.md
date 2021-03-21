# Repo basics

Notebook task.ipynb, and its render, task.html, showcase the analysis.

Brief description of other files in the repo:

- `queries.py` implements queries of The Graph API
- `analysis.py` contains functionality for analysing raw data
- `bitfinex_data.py` contains functionality for parsing csv files with Bitfinex
  prices
- `get_data.py` is a simple script for running queries against The Graph API

As for data files:

- `*.pickle` files contains trades scraped from The Graph API; there are 3
  separate files, as I ran the scraping jobs in three batches.
- `bitfinex_rates/*.csv` contain historical crypto prices from Bitfinex exchange

