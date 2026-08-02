# Advanced Feature Validation

- Rows: 1140
- Generated features: 43
- Missing value counts:
- Infinite value counts:
- Cold-start rows by season:
  - 2023/24: 50
  - 2024/25: 6
  - 2025/26: 5
- Leakage checks:
  - Rolling calculations are based on prior rows only.
  - Current-match values do not contribute to their own features.
  - No future match values enter any rolling statistic.
- Descriptive statistics:
       home_corners  away_corners  total_corners  actual_home_corners  actual_away_corners  actual_total_corners     over_8_5     over_9_5    over_10_5    over_11_5  corners_for_last3  corners_for_last5  corners_for_last10  corners_against_last3  corners_against_last5  corners_against_last10  total_corners_last3  total_corners_last5  total_corners_last10  corners_for_ewma  corners_against_ewma  total_corners_ewma  home_corners_for_last5  home_corners_against_last5  home_total_corners_last5  away_corners_for_last5  away_corners_against_last5  away_total_corners_last5  corners_for_std_last5  corners_for_std_last10  corners_against_std_last5  total_corners_std_last5  total_corners_std_last10  coefficient_of_variation_last10  attack_trend  defence_trend  tempo_trend  expected_home_corners_baseline  expected_away_corners_baseline  expected_total_corners_baseline  attack_difference  defence_difference  tempo_difference  combined_volatility  combined_trend  home_rest_days  away_rest_days  rest_days_difference  home_matches_played  away_matches_played  season_match_number  data_quality_score
count   1140.000000   1140.000000    1140.000000          1140.000000          1140.000000           1140.000000  1140.000000  1140.000000  1140.000000  1140.000000        1140.000000        1140.000000         1140.000000            1140.000000            1140.000000             1140.000000          1140.000000          1140.000000           1140.000000       1140.000000           1140.000000         1140.000000             1140.000000                 1140.000000               1140.000000             1140.000000                 1140.000000               1140.000000            1140.000000             1140.000000                1140.000000              1140.000000               1140.000000                      1140.000000   1140.000000    1140.000000  1140.000000                     1140.000000                     1140.000000                      1140.000000        1140.000000         1140.000000       1140.000000          1140.000000    1.140000e+03     1140.000000     1140.000000           1140.000000          1140.000000          1140.000000          1140.000000         1140.000000
mean       5.211404      4.042982       9.254386             5.211404             4.042982              9.254386     0.560526     0.455263     0.342982     0.233333           4.464620           4.522675            4.551583               4.698977               4.655102                4.647648             9.163596             9.177778              9.199230          4.379786              4.552878            8.932664                4.522675                    4.655102                  9.177778                4.633787                    4.540833                  9.174620               2.205283                2.405179                   2.151154                 2.744747                  2.984041                         0.324952     -0.028907       0.007455    -0.021453                        4.531754                        4.644444                         9.176199          -0.111111            0.114269          0.003158             2.216907   -5.529170e-02        9.091228        8.527193              0.564035            18.500877            18.499123           190.500000            0.941228
std        2.954744      2.480635       3.400437             2.954744             2.480635              3.400437     0.496541     0.498213     0.474914     0.423138           1.672062           1.400347            1.181897               1.830545               1.563996                1.336590             2.216518             1.854275              1.572628          1.378333              1.509861            1.962061                1.400347                    1.563996                  1.854275                1.468798                    1.541235                  1.929686               0.957115                0.805305                   0.952182                 1.132497                  0.921299                         0.100659      0.777494       0.814319     0.993399                        1.082436                        1.089082                         1.472130           1.908425            2.069382          2.378105             0.719655    1.124574e+00       16.890457        9.832047             12.233761            10.970429            10.970909           109.744315            0.193535
min        0.000000      0.000000       1.000000             0.000000             0.000000              1.000000     0.000000     0.000000     0.000000     0.000000           0.000000           0.000000            0.000000               0.000000               0.000000                0.000000             0.000000             0.000000              0.000000          0.000000              0.000000            0.000000                0.000000                    0.000000                  0.000000                0.000000                    0.000000                  0.000000               0.000000                0.000000                   0.000000                 0.000000                  0.000000                         0.000000     -2.800000      -2.400000    -3.600000                        0.000000                        0.000000                         0.000000          -7.000000           -7.000000         -7.000000             0.000000   -4.300000e+00        0.000000        0.000000            -76.000000             0.000000             0.000000             1.000000            0.000000
25%        3.000000      2.000000       7.000000             3.000000             2.000000              7.000000     0.000000     0.000000     0.000000     0.000000           3.333333           3.600000            3.800000               3.333333               3.600000                3.800000             7.666667             8.000000              8.400000          3.495759              3.571220            7.887182                3.600000                    3.600000                  8.000000                3.600000                    3.400000                  8.000000               1.600000                1.920937                   1.549193                 2.000000                  2.459166                         0.269142     -0.500000      -0.500000    -0.700000                        3.800000                        4.000000                         8.400000          -1.400000           -1.200000         -1.600000             1.764313   -7.000000e-01        6.000000        6.000000             -1.000000             9.000000             9.000000            95.750000            1.000000
50%        5.000000      4.000000       9.000000             5.000000             4.000000              9.000000     1.000000     0.000000     0.000000     0.000000           4.333333           4.400000            4.500000               4.666667               4.600000                4.600000             9.333333             9.200000              9.300000          4.357959              4.483748            9.069948                4.400000                    4.600000                  9.200000                4.600000                    4.400000                  9.200000               2.135416                2.357965                   2.059126                 2.653300                  2.993326                         0.323789      0.000000       0.000000     0.000000                        4.500000                        4.600000                         9.200000          -0.166667            0.000000          0.000000             2.188876   -1.110223e-16        7.000000        7.000000              0.000000            19.000000            18.000000           190.500000            1.000000
75%        7.000000      6.000000      11.000000             7.000000             6.000000             11.000000     1.000000     1.000000     1.000000     0.000000           5.333333           5.400000            5.300000               5.666667               5.600000                5.500000            10.666667            10.400000             10.200000          5.301914              5.483835           10.203919                5.400000                    5.600000                 10.400000                5.600000                    5.600000                 10.400000               2.756810                2.864001                   2.611928                 3.328180                  3.515679                         0.388254      0.441667       0.500000     0.600000                        5.200000                        5.300000                        10.100000           1.200000            1.400000          1.400000             2.669036    6.000000e-01        8.000000        8.000000              1.000000            28.000000            28.000000           285.250000            1.000000
max       18.000000     18.000000      21.000000            18.000000            18.000000             21.000000     1.000000     1.000000     1.000000     1.000000          12.000000          10.500000           10.500000              12.000000              10.000000               10.000000            15.666667            15.000000             15.000000         10.046276             10.238602           14.117644               10.500000                   10.000000                 15.000000                9.200000                   10.400000                 14.800000               6.280127                5.431390                   5.851496                 6.711185                  5.513620                         0.688091      3.500000       2.600000     3.300000                        7.800000                        8.800000                        13.900000           6.500000            6.800000         11.600000             4.999392    3.900000e+00      454.000000       92.000000            362.000000            37.000000            37.000000           380.000000            1.000000
- Pearson correlation with total corners:
total_corners                      1.000000
over_9_5                           0.802213
over_10_5                          0.789684
over_8_5                           0.789042
over_11_5                          0.753168
home_corners                       0.703660
actual_home_corners                0.703660
away_corners                       0.532647
actual_away_corners                0.532647
away_corners_against_last5         0.103321
away_total_corners_last5           0.065449
expected_home_corners_baseline     0.064216
defence_trend                      0.058425
expected_total_corners_baseline    0.056421
total_corners_std_last5            0.049623
corners_against_last3              0.039441
home_corners_against_last5         0.038391
corners_against_last5              0.038391
corners_against_std_last5          0.034020
corners_against_ewma               0.032405
total_corners_last10               0.024830
corners_for_last10                 0.022491
home_total_corners_last5           0.021475
total_corners_last5                0.021475
total_corners_std_last10           0.019090
total_corners_ewma                 0.019064
expected_away_corners_baseline     0.012441
corners_against_last10             0.009327
total_corners_last3                0.008976
attack_difference                  0.006666
tempo_trend                        0.000777
coefficient_of_variation_last10    0.000391
away_matches_played                0.000171
home_matches_played               -0.000077
season_match_number               -0.001245
data_quality_score                -0.002209
corners_for_ewma                  -0.008359
corners_for_std_last10            -0.010882
home_corners_for_last5            -0.014441
corners_for_last5                 -0.014441
corners_for_std_last5             -0.021931
away_corners_for_last5            -0.022430
corners_for_last3                 -0.031280
away_rest_days                    -0.031430
tempo_difference                  -0.036364
combined_trend                    -0.037221
combined_volatility               -0.039351
defence_difference                -0.047936
rest_days_difference              -0.050938
home_rest_days                    -0.055190
attack_trend                      -0.060200
- Spearman correlation with total corners:
total_corners                      1.000000
over_9_5                           0.866127
over_8_5                           0.863219
over_10_5                          0.825622
over_11_5                          0.735611
home_corners                       0.678505
actual_home_corners                0.678505
away_corners                       0.505872
actual_away_corners                0.505872
away_corners_against_last5         0.117008
away_total_corners_last5           0.089320
expected_total_corners_baseline    0.081904
expected_home_corners_baseline     0.078352
defence_trend                      0.068292
total_corners_std_last5            0.053161
total_corners_last10               0.040349
home_total_corners_last5           0.038164
total_corners_last5                0.038164
corners_against_last5              0.036028
home_corners_against_last5         0.036028
corners_for_last10                 0.035942
total_corners_std_last10           0.028460
home_rest_days                     0.027818
rest_days_difference               0.025798
corners_against_std_last5          0.025617
corners_against_last3              0.024984
total_corners_ewma                 0.024752
corners_against_ewma               0.019939
coefficient_of_variation_last10    0.016129
expected_away_corners_baseline     0.013542
corners_for_std_last10             0.006636
away_rest_days                     0.004702
total_corners_last3                0.004132
attack_difference                  0.003956
away_matches_played                0.001669
home_matches_played                0.001309
corners_against_last10             0.000509
corners_for_last5                  0.000355
home_corners_for_last5             0.000355
season_match_number                0.000132
corners_for_ewma                  -0.000153
tempo_trend                       -0.003362
data_quality_score                -0.004348
corners_for_std_last5             -0.006336
away_corners_for_last5            -0.020880
combined_volatility               -0.027311
corners_for_last3                 -0.027847
combined_trend                    -0.038153
tempo_difference                  -0.040035
attack_trend                      -0.055221
defence_difference                -0.060168
- Top 15 positive correlations:
total_corners                      1.000000
over_9_5                           0.802213
over_10_5                          0.789684
over_8_5                           0.789042
over_11_5                          0.753168
home_corners                       0.703660
actual_home_corners                0.703660
away_corners                       0.532647
actual_away_corners                0.532647
away_corners_against_last5         0.103321
away_total_corners_last5           0.065449
expected_home_corners_baseline     0.064216
defence_trend                      0.058425
expected_total_corners_baseline    0.056421
total_corners_std_last5            0.049623
- Top 15 negative correlations:
corners_for_ewma         -0.008359
corners_for_std_last10   -0.010882
home_corners_for_last5   -0.014441
corners_for_last5        -0.014441
corners_for_std_last5    -0.021931
away_corners_for_last5   -0.022430
corners_for_last3        -0.031280
away_rest_days           -0.031430
tempo_difference         -0.036364
combined_trend           -0.037221
combined_volatility      -0.039351
defence_difference       -0.047936
rest_days_difference     -0.050938
home_rest_days           -0.055190
attack_trend             -0.060200
- Highly collinear feature pairs above 0.90:
  - corners_for_last5 / home_corners_for_last5: 1.000
  - corners_against_last5 / home_corners_against_last5: 1.000
  - home_total_corners_last5 / total_corners_last5: 1.000
  - away_matches_played / home_matches_played: 1.000
  - away_matches_played / season_match_number: 0.999
  - home_matches_played / season_match_number: 0.999

