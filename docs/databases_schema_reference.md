# Retail Databases Schema Reference

## Overview
This document lists the currently defined schemas for three databases used in the retail time series project:

- **Database 1:** DG
- **Database 2:** S&F
- **Database 3:** WG

---

## Database 1 (DG)

| Column | Definition |
|---|---|
| DATE | `{"type":"DATE","nullable":true}` |
| STORE | `{"type":"FIXED","precision":38,"scale":0,"nullable":true}` |
| CITY | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| STATE | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| OPEN_CLOSED_CODE | `{"type":"TEXT","length":16777216,"byteLength":16777216,"nullable":true,"fixed":false}` |
| UPC | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| SKU | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| P_SKU | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| DESCRIPTION | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| CLASS_NBR | `{"type":"FIXED","precision":19,"scale":0,"nullable":true}` |
| CLASS_DESC | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| DEPARTMENT_NBR | `{"type":"FIXED","precision":19,"scale":0,"nullable":true}` |
| DEPARTMENT_DESC | `{"type":"TEXT","length":2000,"byteLength":8000,"nullable":true,"fixed":false}` |
| TRANS_CT | `{"type":"FIXED","precision":18,"scale":0,"nullable":true}` |
| QTY_SOLD | `{"type":"FIXED","precision":38,"scale":0,"nullable":true}` |
| NET_SALES | `{"type":"FIXED","precision":38,"scale":2,"nullable":true}` |

---

## Database 2 (S&F)

| Column | Definition |
|---|---|
| TRANSACTIONDATE | `NUMBER(38,0)` |
| TRANSACTIONKEY | `VARCHAR(16777216)` |
| POSCARDNUMBER | `VARCHAR(16777216)` |
| STORENUMBER | `NUMBER(38,0)` |
| UPCNUMBER | `NUMBER(38,0)` |
| ITEMNUMBER | `VARCHAR(16777216)` |
| ITEMTOTALSALESPRICEAMOUNT | `NUMBER(38,5)` |
| ITEMQUANTITY | `NUMBER(38,0)` |
| TOTALNETPRICEAMOUNT | `NUMBER(38,5)` |
| FILENAME | `VARCHAR(16777216)` |
| FILEDATE | `NUMBER(38,0)` |
| ROWNUMBER | `NUMBER(38,0)` |

---

## Database 3 (WG)

| Column | Definition |
|---|---|
| DATE | `DATE` |
| STORE | `NUMBER(38,0)` |
| CATEGORY | `VARCHAR(16777216)` |
| CAT_QTY_SOLD | `NUMBER(38,0)` |
| CAT_NET_SALES | `NUMBER(38,5)` |

---

## Common Fields for Harmonization

- **Date:** `DATE`, `TRANSACTIONDATE`
- **Store:** `STORE`, `STORENUMBER`
- **Product:** `UPC`, `UPCNUMBER`, `SKU`, `ITEMNUMBER`
- **Sales:** `NET_SALES`, `ITEMTOTALSALESPRICEAMOUNT`, `TOTALNETPRICEAMOUNT`, `CAT_NET_SALES`
- **Quantity:** `QTY_SOLD`, `ITEMQUANTITY`, `CAT_QTY_SOLD`
