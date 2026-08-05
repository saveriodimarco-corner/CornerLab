# Data Provenance Verification

This report verifies the source provenance of the historical Serie A dataset against the canonical SQLite database and the original Football-Data export.

## 2023/24
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/mmz4281/2324/I1.csv
- Source row count: 380
- File SHA256: e7c23302a33ac86e8aefa0cd0b909a3df033dde0f865235d01dcd50151571ec7
- Import timestamp: 2026-08-05 07:37:02
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## 2024/25
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/mmz4281/2425/I1.csv
- Source row count: 380
- File SHA256: e7c23302a33ac86e8aefa0cd0b909a3df033dde0f865235d01dcd50151571ec7
- Import timestamp: 2026-08-05 07:37:02
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## 2025/26
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/mmz4281/2526/I1.csv
- Source row count: 380
- File SHA256: e7c23302a33ac86e8aefa0cd0b909a3df033dde0f865235d01dcd50151571ec7
- Import timestamp: 2026-08-05 07:37:02
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## Random sample comparisons

### 2023/24

 fixture_id       date   home_team   away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
        267 2024-03-03   Frosinone       Lecce            12             8 2024-03-03    Frosinone        Lecce              12               8
        262 2024-03-02       Monza        Roma             6             3 2024-03-02        Monza         Roma               6               3
        266 2024-03-03      Empoli    Cagliari             9             2 2024-03-03       Empoli     Cagliari               9               2
         40 2023-09-18      Verona     Bologna             3             4 2023-09-18       Verona      Bologna               3               4
         34 2023-09-17    Cagliari     Udinese             4             6 2023-09-17     Cagliari      Udinese               4               6
        364 2024-05-19       Monza   Frosinone             3             4 2024-05-19        Monza    Frosinone               3               4
         56 2023-09-27      Napoli     Udinese             8             4 2023-09-27       Napoli      Udinese               8               4
        263 2024-03-02      Torino  Fiorentina             4             5 2024-03-02       Torino   Fiorentina               4               5
        312 2024-04-13       Lecce      Empoli            10             3 2024-04-13        Lecce       Empoli              10               3
        169 2023-12-23      Torino     Udinese             8             2 2023-12-23       Torino      Udinese               8               2
        235 2024-02-11       Monza      Verona             5             2 2024-02-11        Monza       Verona               5               2
        286 2024-03-17       Inter      Napoli             5             7 2024-03-17        Inter       Napoli               5               7
        142 2023-12-09    Atalanta       Milan             6             4 2023-12-09     Atalanta        Milan               6               4
        251 2024-02-24 Salernitana       Monza             7            11 2024-02-24  Salernitana        Monza               7              11
        117 2023-11-12       Inter   Frosinone             9             4 2023-11-12        Inter    Frosinone               9               4
        359 2024-05-13       Lecce     Udinese             8             2 2024-05-13        Lecce      Udinese               8               2
        243 2024-02-18      Empoli  Fiorentina             0             3 2024-02-18       Empoli   Fiorentina               0               3
         57 2023-09-27      Verona    Atalanta             4             1 2023-09-27       Verona     Atalanta               4               1
        233 2024-02-11       Genoa    Atalanta             7             4 2024-02-11        Genoa     Atalanta               7               4
         91 2023-10-27       Genoa Salernitana             7             0 2023-10-27        Genoa  Salernitana               7               0
         16 2023-08-27    Juventus     Bologna             5             1 2023-08-27     Juventus      Bologna               5               1
        247 2024-02-18     Udinese    Cagliari             3             4 2024-02-18      Udinese     Cagliari               3               4
        375 2024-05-26      Empoli        Roma             4             5 2024-05-26       Empoli         Roma               4               5
          1 2023-08-19      Empoli      Verona             2             4 2023-08-19       Empoli       Verona               2               4
         79 2023-10-08       Monza Salernitana             7             7 2023-10-08        Monza  Salernitana               7               7
        259 2024-02-28       Inter    Atalanta             4             1 2024-02-28        Inter     Atalanta               4               1
         26 2023-09-02     Udinese   Frosinone             3             5 2023-09-02      Udinese    Frosinone               3               5
         47 2023-09-24     Bologna      Napoli             5             5 2023-09-24      Bologna       Napoli               5               5
         10 2023-08-21      Torino    Cagliari             8             3 2023-08-21       Torino     Cagliari               8               3
         77 2023-10-08   Frosinone      Verona             2             4 2023-10-08    Frosinone       Verona               2               4

### 2024/25

 fixture_id       date home_team  away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
        647 2025-03-02     Milan      Lazio             3             5 2025-03-02        Milan        Lazio               3               5
        642 2025-03-01  Atalanta    Venezia             3             2 2025-03-01     Atalanta      Venezia               3               2
        646 2025-03-02     Genoa     Empoli             3             6 2025-03-02        Genoa       Empoli               3               6
        420 2024-09-16     Parma    Udinese             5             6 2024-09-16        Parma      Udinese               5               6
        414 2024-09-15  Atalanta Fiorentina             4             2 2024-09-15     Atalanta   Fiorentina               4               2
        744 2025-05-18     Inter      Lazio             5             1 2025-05-18        Inter        Lazio               5               1
        436 2024-09-29    Empoli Fiorentina             2             4 2024-09-29       Empoli   Fiorentina               2               4
        643 2025-03-01    Napoli      Inter            12             3 2025-03-01       Napoli        Inter              12               3
        692 2025-04-12     Inter   Cagliari             5             6 2025-04-12        Inter     Cagliari               5               6
        549 2024-12-28  Cagliari      Inter             5             7 2024-12-28     Cagliari        Inter               5               7
        615 2025-02-09     Lazio      Monza             8             1 2025-02-09        Lazio        Monza               8               1
        666 2025-03-16  Atalanta      Inter             2             6 2025-03-16     Atalanta        Inter               2               6
        522 2024-12-07  Juventus    Bologna             1             4 2024-12-07     Juventus      Bologna               1               4
        631 2025-02-22     Inter      Genoa             6             7 2025-02-22        Inter        Genoa               6               7
        497 2024-11-10     Inter     Napoli             4             2 2024-11-10        Inter       Napoli               4               2
        739 2025-05-12  Atalanta       Roma             8             7 2025-05-12     Atalanta         Roma               8               7
        623 2025-02-15     Milan     Verona            10             7 2025-02-15        Milan       Verona              10               7
        437 2024-09-29    Napoli      Monza             1             3 2024-09-29       Napoli        Monza               1               3
        613 2025-02-08    Verona   Atalanta             7             2 2025-02-08       Verona     Atalanta               7               2
        471 2024-10-29     Lecce     Verona             3             4 2024-10-29        Lecce       Verona               3               4
        396 2024-08-25    Napoli    Bologna             7             5 2024-08-25       Napoli      Bologna               7               5
        627 2025-02-16     Parma       Roma             1            10 2025-02-16        Parma         Roma               1              10
        755 2025-05-25  Atalanta      Parma             5             6 2025-05-25     Atalanta        Parma               5               6
        381 2024-08-17    Empoli      Monza             7             3 2024-08-17       Empoli        Monza               7               3
        459 2024-10-20   Venezia   Atalanta             2             5 2024-10-20      Venezia     Atalanta               2               5
        639 2025-02-24      Roma      Monza             6             3 2025-02-24         Roma        Monza               6               3
        406 2024-08-31    Napoli      Parma             6             4 2024-08-31       Napoli        Parma               6               4
        427 2024-09-22     Inter      Milan             6             2 2024-09-22        Inter        Milan               6               2
        390 2024-08-19     Lecce   Atalanta             5             3 2024-08-19        Lecce     Atalanta               5               3
        457 2024-10-20     Lecce Fiorentina             1             3 2024-10-20        Lecce   Fiorentina               1               3

### 2025/26

 fixture_id       date  home_team away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
       1027 2026-03-01   Sassuolo  Atalanta             3             9 2026-03-01     Sassuolo     Atalanta               3               9
       1022 2026-02-28       Como     Lecce             5             1 2026-02-28         Como        Lecce               5               1
       1026 2026-03-01       Roma  Juventus             4             1 2026-03-01         Roma     Juventus               4               1
        800 2025-09-22     Napoli      Pisa             5             3 2025-09-22       Napoli         Pisa               5               3
        794 2025-09-20     Verona  Juventus             5             3 2025-09-20       Verona     Juventus               5               3
       1124 2026-05-17      Genoa     Milan             4             4 2026-05-17        Genoa        Milan               4               4
        816 2025-10-05    Bologna      Pisa             2             2 2025-10-05      Bologna         Pisa               2               2
       1023 2026-02-28      Inter     Genoa            10             2 2026-02-28        Inter        Genoa              10               2
       1072 2026-04-11   Atalanta  Juventus            13             2 2026-04-11     Atalanta     Juventus              13               2
        929 2026-01-03       Como   Udinese             3             2 2026-01-03         Como      Udinese               3               2
        995 2026-02-08   Juventus     Lazio             8             1 2026-02-08     Juventus        Lazio               8               1
       1046 2026-03-15      Lazio     Milan             1             9 2026-03-15        Lazio        Milan               1               9
        902 2025-12-13   Atalanta  Cagliari             5             4 2025-12-13     Atalanta     Cagliari               5               4
       1011 2026-02-20   Sassuolo    Verona             6             5 2026-02-20     Sassuolo       Verona               6               5
        877 2025-11-23      Lazio     Lecce             8             2 2025-11-23        Lazio        Lecce               8               2
       1119 2026-05-10     Verona      Como             7             7 2026-05-10       Verona         Como               7               7
       1003 2026-02-14      Lazio  Atalanta             5             4 2026-02-14        Lazio     Atalanta               5               4
        817 2025-10-05 Fiorentina      Roma             3             4 2025-10-05   Fiorentina         Roma               3               4
        993 2026-02-07      Genoa    Napoli             4             3 2026-02-07        Genoa       Napoli               4               3
        851 2025-11-01  Cremonese  Juventus             2             3 2025-11-01    Cremonese     Juventus               2               3
        776 2025-08-30       Pisa      Roma             1             7 2025-08-30         Pisa         Roma               1               7
       1007 2026-02-15     Torino   Bologna             4             2 2026-02-15       Torino      Bologna               4               2
       1135 2026-05-24      Lecce     Genoa             1             6 2026-05-24        Lecce        Genoa               1               6
        761 2025-08-23      Genoa     Lecce             3             7 2025-08-23        Genoa        Lecce               3               7
        839 2025-10-26     Torino     Genoa             9             1 2025-10-26       Torino        Genoa               9               1
       1019 2026-02-23    Bologna   Udinese             4             7 2026-02-23      Bologna      Udinese               4               7
        786 2025-09-14       Pisa   Udinese             8             6 2025-09-14         Pisa      Udinese               8               6
        807 2025-09-28       Roma    Verona             8             1 2025-09-28         Roma       Verona               8               1
        770 2025-08-25    Udinese    Verona             5             1 2025-08-25      Udinese       Verona               5               1
        837 2025-10-26      Lazio  Juventus             2             5 2025-10-26        Lazio     Juventus               2               5
