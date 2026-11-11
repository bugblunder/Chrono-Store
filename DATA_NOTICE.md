# Dataset attribution

The bundled processed SQLite database and training feature CSV derive from:

Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33
https://archive.ics.uci.edu/dataset/352/online+retail

The original data is licensed under Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/

Changes: rows with incomplete customer identity, cancellations and nonpositive purchase lines are excluded; eligible lines are aggregated into invoices; invoice time is the latest eligible line timestamp; naive times are interpreted as Europe/London and converted to UTC; purchase history is transformed into point-in-time features and repeat-purchase labels. Exact counts and checksums are in the import report. No endorsement by the dataset creator is implied.

The synthetic demo generator produces artificial data independently of the UCI workbook.
