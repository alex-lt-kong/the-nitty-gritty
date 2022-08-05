Medium-sized tables (up to 100 million rows) are best stored on disk splayed: each column is stored as a separate file, rather than using a single file for the whole table.

Reference: https://code.kx.com/q/kb/splayed-tables/tr