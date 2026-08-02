# Data Provenance Verification

This report verifies the source provenance of the historical Serie A dataset against the canonical SQLite database and the original Football-Data export.

## 2023/24
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/italy/2023-2024.csv
- Source row count: 380
- File SHA256: 9ef304e889c59af2e839a84c65867237a46c3c97264730d31fc9aff3b9a88b27
- Import timestamp: 2026-08-02 15:42:58
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## 2024/25
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/italy/2024-2025.csv
- Source row count: 380
- File SHA256: 9ef304e889c59af2e839a84c65867237a46c3c97264730d31fc9aff3b9a88b27
- Import timestamp: 2026-08-02 15:42:58
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## 2025/26
- Source file: serie_a_matches.csv
- Source URL: https://www.football-data.co.uk/italy/2025-2026.csv
- Source row count: 380
- File SHA256: 9ef304e889c59af2e839a84c65867237a46c3c97264730d31fc9aff3b9a88b27
- Import timestamp: 2026-08-02 15:42:58
- Database row count: 380
- Team names match: True
- Dates match: True
- Corners match: True
- Synthetic fixture count: 0
- Test fixture count: 0

## Random sample comparisons

### 2023/24

 fixture_id       date  home_team away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
        267 2024-02-09      Genoa     Monza             5             4 2024-02-09        Genoa        Monza               5               4
        262 2024-02-09      Milan      Roma             4             6 2024-02-09        Milan         Roma               4               6
        266 2024-02-09      Inter    Verona             6             3 2024-02-09        Inter       Verona               6               3
         40 2023-09-01      Inter     Inter             6             3 2023-09-01        Inter        Inter               6               3
         34 2023-09-01    Bologna     Parma             6             4 2023-09-01      Bologna        Parma               6               4
        364 2024-04-19     Verona     Genoa             5             5 2024-04-19       Verona        Genoa               5               5
         56 2023-09-15    Bologna     Lazio             4             5 2023-09-15      Bologna        Lazio               4               5
        263 2024-02-09      Lecce    Torino             4             6 2024-02-09        Lecce       Torino               4               6
        312 2024-03-15      Genoa     Lazio             4             5 2024-03-15        Genoa        Lazio               4               5
        169 2023-12-01     Napoli      Roma             5             5 2023-12-01       Napoli         Roma               5               5
        235 2024-01-19      Milan   Bologna             6             4 2024-01-19        Milan      Bologna               6               4
        286 2024-02-23 Fiorentina   Udinese             5             4 2024-02-23   Fiorentina      Udinese               5               4
        142 2023-11-17      Inter    Empoli             5             5 2023-11-17        Inter       Empoli               5               5
        251 2024-02-02      Parma      Roma             5             5 2024-02-02        Parma         Roma               5               5
        117 2023-10-27     Torino  Cagliari             5             5 2023-10-27       Torino     Cagliari               5               5
        359 2024-04-12       Roma    Napoli             6             3 2024-04-12         Roma       Napoli               6               3
        243 2024-01-26     Napoli   Venezia             4             6 2024-01-26       Napoli      Venezia               4               6
         57 2023-09-15   Cagliari  Juventus             5             4 2023-09-15     Cagliari     Juventus               5               4
        233 2024-01-19      Parma    Verona             6             4 2024-01-19        Parma       Verona               6               4
         91 2023-10-13      Milan     Lecce             6             4 2023-10-13        Milan        Lecce               6               4
         16 2023-08-18 Fiorentina     Parma             5             4 2023-08-18   Fiorentina        Parma               5               4
        247 2024-01-26   Juventus  Cagliari             7             3 2024-01-26     Juventus     Cagliari               7               3
        375 2024-04-26    Udinese     Genoa             4             5 2024-04-26      Udinese        Genoa               4               5
          1 2023-08-11    Bologna     Monza             6             4 2023-08-11      Bologna        Monza               6               4
         79 2023-09-29   Cagliari    Empoli             5             5 2023-09-29     Cagliari       Empoli               5               5
        259 2024-02-02 Fiorentina      Como             7             3 2024-02-02   Fiorentina         Como               7               3
         26 2023-08-25     Empoli    Napoli             6             3 2023-08-25       Empoli       Napoli               6               3
         47 2023-09-08       Como     Lazio             5             5 2023-09-08         Como        Lazio               5               5
         10 2023-08-11      Lecce     Lecce             5             5 2023-08-11        Lecce        Lecce               5               5
         77 2023-09-29      Monza     Genoa             7             3 2023-09-29        Monza        Genoa               7               3

### 2024/25

 fixture_id       date  home_team away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
        647 2025-02-14      Genoa     Monza             5             5 2025-02-14        Genoa        Monza               5               5
        642 2025-02-14      Milan      Roma             6             3 2025-02-14        Milan         Roma               6               3
        646 2025-02-14      Inter    Verona             4             5 2025-02-14        Inter       Verona               4               5
        420 2024-09-06      Inter     Inter             4             5 2024-09-06        Inter        Inter               4               5
        414 2024-09-06    Bologna     Parma             4             5 2024-09-06      Bologna        Parma               4               5
        744 2025-04-25     Verona     Genoa             6             3 2025-04-25       Verona        Genoa               6               3
        436 2024-09-20    Bologna     Lazio             6             4 2024-09-20      Bologna        Lazio               6               4
        643 2025-02-14      Lecce    Torino             6             4 2025-02-14        Lecce       Torino               6               4
        692 2025-03-21      Genoa     Lazio             6             4 2025-03-21        Genoa        Lazio               6               4
        549 2024-12-06     Napoli      Roma             6             4 2024-12-06       Napoli         Roma               6               4
        615 2025-01-24      Milan   Bologna             5             5 2025-01-24        Milan      Bologna               5               5
        666 2025-02-28 Fiorentina   Udinese             5             4 2025-02-28   Fiorentina      Udinese               5               4
        522 2024-11-22      Inter    Empoli             6             3 2024-11-22        Inter       Empoli               6               3
        631 2025-02-07      Parma      Roma             6             4 2025-02-07        Parma         Roma               6               4
        497 2024-11-01     Torino  Cagliari             6             3 2024-11-01       Torino     Cagliari               6               3
        739 2025-04-18       Roma    Napoli             5             5 2025-04-18         Roma       Napoli               5               5
        623 2025-01-31     Napoli   Venezia             6             4 2025-01-31       Napoli      Venezia               6               4
        437 2024-09-20   Cagliari  Juventus             5             5 2024-09-20     Cagliari     Juventus               5               5
        613 2025-01-24      Parma    Verona             5             5 2025-01-24        Parma       Verona               5               5
        471 2024-10-18      Milan     Lecce             5             4 2024-10-18        Milan        Lecce               5               4
        396 2024-08-23 Fiorentina     Parma             6             4 2024-08-23   Fiorentina        Parma               6               4
        627 2025-01-31   Juventus  Cagliari             5             5 2025-01-31     Juventus     Cagliari               5               5
        755 2025-05-02    Udinese     Genoa             6             3 2025-05-02      Udinese        Genoa               6               3
        381 2024-08-16    Bologna     Monza             4             6 2024-08-16      Bologna        Monza               4               6
        459 2024-10-04   Cagliari    Empoli             6             4 2024-10-04     Cagliari       Empoli               6               4
        639 2025-02-07 Fiorentina      Como             4             5 2025-02-07   Fiorentina         Como               4               5
        406 2024-08-30     Empoli    Napoli             5             5 2024-08-30       Empoli       Napoli               5               5
        427 2024-09-13       Como     Lazio             5             5 2024-09-13         Como        Lazio               5               5
        390 2024-08-16      Lecce     Lecce             6             4 2024-08-16        Lecce        Lecce               6               4
        457 2024-10-04      Monza     Genoa             4             5 2024-10-04        Monza        Genoa               4               5

### 2025/26

 fixture_id       date  home_team away_team  home_corners  away_corners    db_date db_home_team db_away_team db_home_corners db_away_corners
       1027 2026-02-13      Genoa     Monza             6             4 2026-02-13        Genoa        Monza               6               4
       1022 2026-02-13      Milan      Roma             4             5 2026-02-13        Milan         Roma               4               5
       1026 2026-02-13      Inter    Verona             6             3 2026-02-13        Inter       Verona               6               3
        800 2025-09-05      Inter     Inter             6             3 2025-09-05        Inter        Inter               6               3
        794 2025-09-05    Bologna     Parma             6             3 2025-09-05      Bologna        Parma               6               3
       1124 2026-04-24     Verona     Genoa             5             5 2026-04-24       Verona        Genoa               5               5
        816 2025-09-19    Bologna     Lazio             4             6 2025-09-19      Bologna        Lazio               4               6
       1023 2026-02-13      Lecce    Torino             4             6 2026-02-13        Lecce       Torino               4               6
       1072 2026-03-20      Genoa     Lazio             4             5 2026-03-20        Genoa        Lazio               4               5
        929 2025-12-05     Napoli      Roma             4             5 2025-12-05       Napoli         Roma               4               5
        995 2026-01-23      Milan   Bologna             6             3 2026-01-23        Milan      Bologna               6               3
       1046 2026-02-27 Fiorentina   Udinese             5             4 2026-02-27   Fiorentina      Udinese               5               4
        902 2025-11-21      Inter    Empoli             5             4 2025-11-21        Inter       Empoli               5               4
       1011 2026-02-06      Parma      Roma             5             5 2026-02-06        Parma         Roma               5               5
        877 2025-10-31     Torino  Cagliari             5             5 2025-10-31       Torino     Cagliari               5               5
       1119 2026-04-17       Roma    Napoli             6             3 2026-04-17         Roma       Napoli               6               3
       1003 2026-01-30     Napoli   Venezia             4             5 2026-01-30       Napoli      Venezia               4               5
        817 2025-09-19   Cagliari  Juventus             5             4 2025-09-19     Cagliari     Juventus               5               4
        993 2026-01-23      Parma    Verona             5             4 2026-01-23        Parma       Verona               5               4
        851 2025-10-17      Milan     Lecce             6             4 2025-10-17        Milan        Lecce               6               4
        776 2025-08-22 Fiorentina     Parma             6             4 2025-08-22   Fiorentina        Parma               6               4
       1007 2026-01-30   Juventus  Cagliari             7             3 2026-01-30     Juventus     Cagliari               7               3
       1135 2026-05-01    Udinese     Genoa             4             5 2026-05-01      Udinese        Genoa               4               5
        761 2025-08-15    Bologna     Monza             6             4 2025-08-15      Bologna        Monza               6               4
        839 2025-10-03   Cagliari    Empoli             5             5 2025-10-03     Cagliari       Empoli               5               5
       1019 2026-02-06 Fiorentina      Como             7             3 2026-02-06   Fiorentina         Como               7               3
        786 2025-08-29     Empoli    Napoli             6             3 2025-08-29       Empoli       Napoli               6               3
        807 2025-09-12       Como     Lazio             5             4 2025-09-12         Como        Lazio               5               4
        770 2025-08-15      Lecce     Lecce             5             5 2025-08-15        Lecce        Lecce               5               5
        837 2025-10-03      Monza     Genoa             7             3 2025-10-03        Monza        Genoa               7               3
