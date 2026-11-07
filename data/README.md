# Data

No data is committed to this repository. Run `python scripts/download_data.py` to fetch it.

## Source
- **Dataset:** Online Retail II
- **Creator:** Daqing Chen
- **Host:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **DOI:** 10.24432/C5CG6D
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
- **Attribution:** Dataset: Online Retail II by Daqing Chen, UCI Machine Learning Repository, licensed under CC BY 4.0.

## What it contains
All transactions of a UK-based online retailer of gift and homeware items between 2009-12-01 and 2011-12-09: 1,067,371 invoice lines with 8 columns (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country). The source notes that many customers are wholesalers. There is no marketing contact data.

## Download checks
`scripts/download_data.py` downloads over HTTPS from the UCI host, verifies SHA-256 `572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb`, rejects unsafe archive paths, extracts into an empty folder, checks the file type, and confirms the row and column counts before writing `data/raw/online_retail_II.csv`.

## Privacy
Customer ID is a pseudonymous number. No names, emails, phone numbers or addresses are present.
