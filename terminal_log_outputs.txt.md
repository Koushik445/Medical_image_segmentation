**UNET-Base Training**

(segmentation) D:\\Brain\_Segmentation>python run\_local.py --data\_dir processed\_dataset



============================================================

&#x20; GPU   : NVIDIA GeForce RTX 4060 Laptop GPU

&#x20; VRAM  : 8.0 GB

&#x20; AMP   : autocast + GradScaler enabled

&#x20; cuDNN : benchmark=True

============================================================



============================================================

&#x20; STAGE 1: Standard U-Net (Baseline)

============================================================

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/50] train=0.5417 | val=4.6945 | dice=0.1212 | best=0.1212 | lr=9.99e-04 | VRAM=486MB | 52s

&#x20; Epoch \[  2/50] train=0.3992 | val=0.4987 | dice=0.2501 | best=0.2501 | lr=9.96e-04 | VRAM=486MB | 62s

&#x20; Epoch \[  3/50] train=0.3234 | val=0.4261 | dice=0.2644 | best=0.2644 | lr=9.91e-04 | VRAM=486MB | 73s

&#x20; Epoch \[  4/50] train=0.2880 | val=0.3295 | dice=0.4457 | best=0.4457 | lr=9.84e-04 | VRAM=486MB | 83s

&#x20; Epoch \[  5/50] train=0.2543 | val=0.2541 | dice=0.5950 | best=0.5950 | lr=9.76e-04 | VRAM=486MB | 94s

&#x20; Epoch \[  6/50] train=0.2393 | val=0.2583 | dice=0.5727 | best=0.5950 | lr=9.65e-04 | VRAM=486MB | 104s

&#x20; Epoch \[  7/50] train=0.2222 | val=0.2953 | dice=0.4978 | best=0.5950 | lr=9.52e-04 | VRAM=486MB | 114s

&#x20; Epoch \[  8/50] train=0.2041 | val=0.2072 | dice=0.6648 | best=0.6648 | lr=9.38e-04 | VRAM=486MB | 125s

&#x20; Epoch \[  9/50] train=0.2018 | val=0.2546 | dice=0.6136 | best=0.6648 | lr=9.22e-04 | VRAM=486MB | 135s

&#x20; Epoch \[ 10/50] train=0.1949 | val=0.2034 | dice=0.6694 | best=0.6694 | lr=9.05e-04 | VRAM=486MB | 146s

&#x20; Epoch \[ 11/50] train=0.1768 | val=0.2298 | dice=0.6173 | best=0.6694 | lr=8.85e-04 | VRAM=486MB | 156s

&#x20; Epoch \[ 12/50] train=0.1670 | val=0.1901 | dice=0.6902 | best=0.6902 | lr=8.65e-04 | VRAM=486MB | 167s

&#x20; Epoch \[ 13/50] train=0.1554 | val=0.1621 | dice=0.7326 | best=0.7326 | lr=8.42e-04 | VRAM=486MB | 178s

&#x20; Epoch \[ 14/50] train=0.1516 | val=0.1793 | dice=0.7063 | best=0.7326 | lr=8.19e-04 | VRAM=486MB | 188s

&#x20; Epoch \[ 15/50] train=0.1562 | val=0.1748 | dice=0.7087 | best=0.7326 | lr=7.94e-04 | VRAM=486MB | 199s

&#x20; Epoch \[ 16/50] train=0.1440 | val=0.1624 | dice=0.7328 | best=0.7328 | lr=7.68e-04 | VRAM=486MB | 210s

&#x20; Epoch \[ 17/50] train=0.1481 | val=0.1824 | dice=0.7025 | best=0.7328 | lr=7.41e-04 | VRAM=486MB | 220s

&#x20; Epoch \[ 18/50] train=0.1495 | val=0.1573 | dice=0.7409 | best=0.7409 | lr=7.13e-04 | VRAM=486MB | 231s

&#x20; Epoch \[ 19/50] train=0.1266 | val=0.1650 | dice=0.7257 | best=0.7409 | lr=6.84e-04 | VRAM=486MB | 242s

&#x20; Epoch \[ 20/50] train=0.1233 | val=0.1593 | dice=0.7473 | best=0.7473 | lr=6.55e-04 | VRAM=486MB | 254s

&#x20; Epoch \[ 21/50] train=0.1249 | val=0.1559 | dice=0.7423 | best=0.7473 | lr=6.25e-04 | VRAM=486MB | 265s

&#x20; Epoch \[ 22/50] train=0.1179 | val=0.1421 | dice=0.7626 | best=0.7626 | lr=5.94e-04 | VRAM=486MB | 276s

&#x20; Epoch \[ 23/50] train=0.1242 | val=0.1560 | dice=0.7399 | best=0.7626 | lr=5.63e-04 | VRAM=486MB | 286s

&#x20; Epoch \[ 24/50] train=0.1165 | val=0.1483 | dice=0.7548 | best=0.7626 | lr=5.32e-04 | VRAM=486MB | 296s

&#x20; Epoch \[ 25/50] train=0.1124 | val=0.1251 | dice=0.7924 | best=0.7924 | lr=5.00e-04 | VRAM=486MB | 307s

&#x20; Epoch \[ 26/50] train=0.1048 | val=0.1325 | dice=0.7805 | best=0.7924 | lr=4.69e-04 | VRAM=486MB | 318s

&#x20; Epoch \[ 27/50] train=0.1039 | val=0.1301 | dice=0.7830 | best=0.7924 | lr=4.38e-04 | VRAM=486MB | 329s

&#x20; Epoch \[ 28/50] train=0.0971 | val=0.1205 | dice=0.7983 | best=0.7983 | lr=4.07e-04 | VRAM=486MB | 340s

&#x20; Epoch \[ 29/50] train=0.0950 | val=0.1506 | dice=0.7517 | best=0.7983 | lr=3.76e-04 | VRAM=486MB | 351s

&#x20; Epoch \[ 30/50] train=0.0962 | val=0.1219 | dice=0.7969 | best=0.7983 | lr=3.46e-04 | VRAM=486MB | 362s

&#x20; Epoch \[ 31/50] train=0.0899 | val=0.1187 | dice=0.8000 | best=0.8000 | lr=3.17e-04 | VRAM=486MB | 374s

&#x20; Epoch \[ 32/50] train=0.0884 | val=0.1170 | dice=0.8046 | best=0.8046 | lr=2.88e-04 | VRAM=486MB | 385s

&#x20; Epoch \[ 33/50] train=0.0845 | val=0.1210 | dice=0.7996 | best=0.8046 | lr=2.60e-04 | VRAM=486MB | 396s

&#x20; Epoch \[ 34/50] train=0.0856 | val=0.1145 | dice=0.8098 | best=0.8098 | lr=2.33e-04 | VRAM=486MB | 407s

&#x20; Epoch \[ 35/50] train=0.0831 | val=0.1193 | dice=0.7997 | best=0.8098 | lr=2.07e-04 | VRAM=486MB | 417s

&#x20; Epoch \[ 36/50] train=0.0789 | val=0.1133 | dice=0.8111 | best=0.8111 | lr=1.82e-04 | VRAM=486MB | 428s

&#x20; Epoch \[ 37/50] train=0.0751 | val=0.1121 | dice=0.8118 | best=0.8118 | lr=1.59e-04 | VRAM=486MB | 439s

&#x20; Epoch \[ 38/50] train=0.0730 | val=0.1098 | dice=0.8159 | best=0.8159 | lr=1.36e-04 | VRAM=486MB | 449s

&#x20; Epoch \[ 39/50] train=0.0714 | val=0.1218 | dice=0.7943 | best=0.8159 | lr=1.16e-04 | VRAM=486MB | 460s

&#x20; Epoch \[ 40/50] train=0.0722 | val=0.1096 | dice=0.8166 | best=0.8166 | lr=9.64e-05 | VRAM=486MB | 471s

&#x20; Epoch \[ 41/50] train=0.0694 | val=0.1105 | dice=0.8152 | best=0.8166 | lr=7.88e-05 | VRAM=486MB | 481s

&#x20; Epoch \[ 42/50] train=0.0674 | val=0.1113 | dice=0.8144 | best=0.8166 | lr=6.28e-05 | VRAM=486MB | 493s

&#x20; Epoch \[ 43/50] train=0.0668 | val=0.1085 | dice=0.8167 | best=0.8167 | lr=4.85e-05 | VRAM=486MB | 505s

&#x20; Epoch \[ 44/50] train=0.0655 | val=0.1084 | dice=0.8182 | best=0.8182 | lr=3.61e-05 | VRAM=486MB | 516s

&#x20; Epoch \[ 45/50] train=0.0648 | val=0.1080 | dice=0.8184 | best=0.8184 | lr=2.54e-05 | VRAM=486MB | 527s

&#x20; Epoch \[ 46/50] train=0.0636 | val=0.1084 | dice=0.8181 | best=0.8184 | lr=1.67e-05 | VRAM=486MB | 538s

&#x20; Epoch \[ 47/50] train=0.0630 | val=0.1087 | dice=0.8184 | best=0.8184 | lr=9.85e-06 | VRAM=486MB | 549s

&#x20; Epoch \[ 48/50] train=0.0631 | val=0.1093 | dice=0.8165 | best=0.8184 | lr=4.94e-06 | VRAM=486MB | 559s

&#x20; Epoch \[ 49/50] train=0.0621 | val=0.1096 | dice=0.8163 | best=0.8184 | lr=1.99e-06 | VRAM=486MB | 570s

&#x20; Epoch \[ 50/50] train=0.0619 | val=0.1089 | dice=0.8167 | best=0.8184 | lr=1.00e-06 | VRAM=486MB | 580s



&#x20; ✓ Standard U-Net best Dice: 0.8184





















**PSO-Search and retraining logs**



(segmentation) D:\\Brain\_Segmentation>python train.py --data\_dir processed\_dataset --use\_pso



\[Device] NVIDIA GeForce RTX 4060 Laptop GPU | VRAM: 8.0 GB

\[AMP]    autocast + GradScaler enabled





\[PSO-Hyper] Starting swarm initialization

\[PSO-Hyper] 6 particles × 8 iters = 54 total probe runs



\[PSO-Hyper] Phase 1/2 — initializing all 6 particles ...



==============================================================

&#x20; \[Probe 001/54] lr=0.00381  bs=16

==============================================================



&#x20; \[probe] lr=0.00381, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.4511 | val=0.4264 | dice=0.3549 | best=0.3549 | lr=3.66e-03 | VRAM=485MB | 12s

&#x20; Epoch \[  2/8] train=0.3322 | val=0.4532 | dice=0.3926 | best=0.3926 | lr=3.25e-03 | VRAM=486MB | 22s

&#x20; Epoch \[  3/8] train=0.2937 | val=0.3718 | dice=0.4016 | best=0.4016 | lr=2.63e-03 | VRAM=486MB | 32s





.........

.........

.........

.........

==============================================================



&#x20; \[probe] lr=0.00285, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.4738 | val=0.5474 | dice=0.2877 | best=0.2877 | lr=2.74e-03 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.3465 | val=0.3288 | dice=0.4721 | best=0.4721 | lr=2.43e-03 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.3116 | val=0.3370 | dice=0.4584 | best=0.4721 | lr=1.97e-03 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.2955 | val=0.3852 | dice=0.3580 | best=0.4721 | lr=1.43e-03 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.2741 | val=0.2630 | dice=0.5694 | best=0.5694 | lr=8.81e-04 | VRAM=487MB | 48s

&#x20; Epoch \[  6/8] train=0.2566 | val=0.2558 | dice=0.5848 | best=0.5848 | lr=4.19e-04 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.2416 | val=0.2431 | dice=0.6059 | best=0.6059 | lr=1.10e-04 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.2257 | val=0.2282 | dice=0.6281 | best=0.6281 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.6281 | VRAM=0MB

&#x20; \[Probe 007/54] → dice=0.6281 | VRAM=0MB



==============================================================

&#x20; \[Probe 008/54] lr=0.00278  bs=16

==============================================================



&#x20; \[probe] lr=0.00278, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.4721 | val=0.4235 | dice=0.3417 | best=0.3417 | lr=2.68e-03 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.3403 | val=0.3996 | dice=0.4133 | best=0.4133 | lr=2.38e-03 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.3077 | val=0.3986 | dice=0.2997 | best=0.4133 | lr=1.92e-03 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.2903 | val=0.3169 | dice=0.4656 | best=0.4656 | lr=1.39e-03 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.2632 | val=0.2900 | dice=0.5190 | best=0.5190 | lr=8.60e-04 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.2521 | val=0.2694 | dice=0.5702 | best=0.5702 | lr=4.08e-04 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.2372 | val=0.2518 | dice=0.5990 | best=0.5990 | lr=1.07e-04 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.2222 | val=0.2278 | dice=0.6272 | best=0.6272 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.6272 | VRAM=0MB

&#x20; \[Probe 008/54] → dice=0.6272 | VRAM=0MB



==============================================================

&#x20; \[Probe 009/54] lr=0.00106  bs=16

==============================================================



&#x20; \[probe] lr=0.00106, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5376 | val=0.4649 | dice=0.4022 | best=0.4022 | lr=1.02e-03 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.3972 | val=0.3911 | dice=0.4460 | best=0.4460 | lr=9.02e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.3171 | val=0.3179 | dice=0.4923 | best=0.4923 | lr=7.31e-04 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.2795 | val=0.3138 | dice=0.4873 | best=0.4923 | lr=5.29e-04 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.2561 | val=0.2715 | dice=0.5661 | best=0.5661 | lr=3.27e-04 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.2351 | val=0.2271 | dice=0.6441 | best=0.6441 | lr=1.56e-04 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.2178 | val=0.2214 | dice=0.6541 | best=0.6541 | lr=4.12e-05 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.2034 | val=0.2032 | dice=0.6797 | best=0.6797 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.6797 | VRAM=0MB

&#x20; \[Probe 009/54] → dice=0.6797 | VRAM=0MB



==============================================================

&#x20; \[Probe 010/54] lr=0.00056  bs=16

==============================================================



&#x20; \[probe] lr=0.00056, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5694 | val=0.7531 | dice=0.1869 | best=0.1869 | lr=5.42e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.4506 | val=0.4200 | dice=0.4569 | best=0.4569 | lr=4.81e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.3567 | val=0.3469 | dice=0.5502 | best=0.5502 | lr=3.90e-04 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.2931 | val=0.2869 | dice=0.5931 | best=0.5931 | lr=2.82e-04 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.2536 | val=0.2489 | dice=0.6594 | best=0.6594 | lr=1.75e-04 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.2232 | val=0.2196 | dice=0.6944 | best=0.6944 | lr=8.34e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.2005 | val=0.2015 | dice=0.7205 | best=0.7205 | lr=2.24e-05 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.1877 | val=0.1934 | dice=0.7304 | best=0.7304 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7304 | VRAM=0MB

&#x20; \[Probe 010/54] → dice=0.7304 | VRAM=0MB



==============================================================

&#x20; \[Probe 011/54] lr=0.00081  bs=16

==============================================================



&#x20; \[probe] lr=0.00081, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5549 | val=0.6728 | dice=0.2087 | best=0.2087 | lr=7.83e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.4228 | val=0.3918 | dice=0.4449 | best=0.4449 | lr=6.94e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.3484 | val=0.3701 | dice=0.4648 | best=0.4648 | lr=5.63e-04 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.3091 | val=0.3072 | dice=0.5296 | best=0.5296 | lr=4.07e-04 | VRAM=487MB | 40s

&#x20; Epoch \[  5/8] train=0.2813 | val=0.2769 | dice=0.5650 | best=0.5650 | lr=2.52e-04 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.2594 | val=0.2518 | dice=0.6088 | best=0.6088 | lr=1.20e-04 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.2436 | val=0.2379 | dice=0.6338 | best=0.6338 | lr=3.19e-05 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.2270 | val=0.2331 | dice=0.6352 | best=0.6352 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.6352 | VRAM=0MB

&#x20; \[Probe 011/54] → dice=0.6352 | VRAM=0MB



==============================================================

&#x20; \[Probe 012/54] lr=0.00030  bs=16

==============================================================



&#x20; \[probe] lr=0.00030, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5906 | val=0.5786 | dice=0.3529 | best=0.3529 | lr=2.92e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.4993 | val=0.4750 | dice=0.4548 | best=0.4548 | lr=2.59e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.4406 | val=0.4508 | dice=0.5227 | best=0.5227 | lr=2.10e-04 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.3867 | val=0.3888 | dice=0.5528 | best=0.5528 | lr=1.52e-04 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.3444 | val=0.3429 | dice=0.6410 | best=0.6410 | lr=9.45e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.3157 | val=0.3036 | dice=0.7120 | best=0.7120 | lr=4.53e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.2943 | val=0.2935 | dice=0.7201 | best=0.7201 | lr=1.25e-05 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.2820 | val=0.2861 | dice=0.7302 | best=0.7302 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7302 | VRAM=0MB

&#x20; \[Probe 012/54] → dice=0.7302 | VRAM=0MB



&#x20; \[Iter  1/8 SUMMARY] w=0.900 | best\_dice=0.7384 | lr=0.00030, bs=16



\[PSO-Hyper] ── Iter  2/8 (w=0.838) ──────────────────────



==============================================================

&#x20; \[Probe 013/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6163 | val=0.5756 | dice=0.3365 | best=0.3365 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5478 | val=0.5515 | dice=0.4584 | best=0.4584 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5112 | val=0.4974 | dice=0.6362 | best=0.6362 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4841 | val=0.4775 | dice=0.6521 | best=0.6521 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4630 | val=0.4654 | dice=0.6632 | best=0.6632 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4496 | val=0.4507 | dice=0.7285 | best=0.7285 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4402 | val=0.4424 | dice=0.7306 | best=0.7306 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4354 | val=0.4424 | dice=0.7370 | best=0.7370 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7370 | VRAM=0MB

&#x20; \[Probe 013/54] → dice=0.7370 | VRAM=0MB



==============================================================

&#x20; \[Probe 014/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6170 | val=0.5750 | dice=0.4440 | best=0.4440 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5423 | val=0.5490 | dice=0.4573 | best=0.4573 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5127 | val=0.5034 | dice=0.6275 | best=0.6275 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4842 | val=0.4867 | dice=0.6267 | best=0.6275 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4630 | val=0.4632 | dice=0.6910 | best=0.6910 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4488 | val=0.4504 | dice=0.7251 | best=0.7251 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4404 | val=0.4428 | dice=0.7316 | best=0.7316 | lr=4.77e-06 | VRAM=487MB | 70s

&#x20; Epoch \[  8/8] train=0.4353 | val=0.4409 | dice=0.7405 | best=0.7405 | lr=1.00e-06 | VRAM=487MB | 82s

→ dice=0.7405 | VRAM=0MB

&#x20; \[Probe 014/54] → dice=0.7405 | VRAM=0MB



==============================================================

&#x20; \[Probe 015/54] lr=0.00012  bs=16

==============================================================



&#x20; \[probe] lr=0.00012, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6118 | val=0.5818 | dice=0.4390 | best=0.4390 | lr=1.13e-04 | VRAM=487MB | 12s

&#x20; Epoch \[  2/8] train=0.5370 | val=0.5716 | dice=0.4189 | best=0.4390 | lr=1.00e-04 | VRAM=487MB | 23s

&#x20; Epoch \[  3/8] train=0.5035 | val=0.6063 | dice=0.3977 | best=0.4390 | lr=8.15e-05 | VRAM=487MB | 35s

&#x20; Epoch \[  4/8] train=0.4749 | val=0.4678 | dice=0.6703 | best=0.6703 | lr=5.92e-05 | VRAM=487MB | 46s

&#x20; Epoch \[  5/8] train=0.4494 | val=0.4482 | dice=0.6820 | best=0.6820 | lr=3.69e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  6/8] train=0.4338 | val=0.4348 | dice=0.7258 | best=0.7258 | lr=1.81e-05 | VRAM=487MB | 68s

&#x20; Epoch \[  7/8] train=0.4243 | val=0.4276 | dice=0.7305 | best=0.7305 | lr=5.43e-06 | VRAM=487MB | 80s

&#x20; Epoch \[  8/8] train=0.4191 | val=0.4246 | dice=0.7440 | best=0.7440 | lr=1.00e-06 | VRAM=487MB | 91s

→ dice=0.7440 | VRAM=0MB

&#x20; \[Probe 015/54] → dice=0.7440 | VRAM=0MB



==============================================================

&#x20; \[Probe 016/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6149 | val=0.5673 | dice=0.3691 | best=0.3691 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5428 | val=0.5359 | dice=0.5220 | best=0.5220 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5095 | val=0.4999 | dice=0.6334 | best=0.6334 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4837 | val=0.4803 | dice=0.6656 | best=0.6656 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4614 | val=0.4593 | dice=0.6772 | best=0.6772 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4485 | val=0.4457 | dice=0.7220 | best=0.7220 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4392 | val=0.4414 | dice=0.7278 | best=0.7278 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4343 | val=0.4392 | dice=0.7387 | best=0.7387 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7387 | VRAM=0MB

&#x20; \[Probe 016/54] → dice=0.7387 | VRAM=0MB



==============================================================

&#x20; \[Probe 017/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6149 | val=0.5754 | dice=0.4509 | best=0.4509 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5417 | val=0.5608 | dice=0.4442 | best=0.4509 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5101 | val=0.5023 | dice=0.6394 | best=0.6394 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4846 | val=0.4856 | dice=0.6287 | best=0.6394 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4659 | val=0.4631 | dice=0.6722 | best=0.6722 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4496 | val=0.4524 | dice=0.7209 | best=0.7209 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4406 | val=0.4427 | dice=0.7297 | best=0.7297 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4358 | val=0.4412 | dice=0.7405 | best=0.7405 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7405 | VRAM=0MB

&#x20; \[Probe 017/54] → dice=0.7405 | VRAM=0MB



==============================================================

&#x20; \[Probe 018/54] lr=0.00029  bs=16

==============================================================



&#x20; \[probe] lr=0.00029, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5917 | val=0.5707 | dice=0.3719 | best=0.3719 | lr=2.80e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.4957 | val=0.4937 | dice=0.4490 | best=0.4490 | lr=2.49e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.4352 | val=0.4227 | dice=0.6105 | best=0.6105 | lr=2.02e-04 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.3845 | val=0.3687 | dice=0.6353 | best=0.6353 | lr=1.46e-04 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.3444 | val=0.3456 | dice=0.6223 | best=0.6353 | lr=9.06e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.3150 | val=0.3133 | dice=0.7050 | best=0.7050 | lr=4.35e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.2960 | val=0.2955 | dice=0.7347 | best=0.7347 | lr=1.20e-05 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.2847 | val=0.2915 | dice=0.7421 | best=0.7421 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7421 | VRAM=0MB

&#x20; \[Probe 018/54] → dice=0.7421 | VRAM=0MB



&#x20; \[Iter  2/8 SUMMARY] w=0.838 | best\_dice=0.7440 | lr=0.00012, bs=16



\[PSO-Hyper] ── Iter  3/8 (w=0.775) ──────────────────────



==============================================================

&#x20; \[Probe 019/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6165 | val=0.5733 | dice=0.4725 | best=0.4725 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5423 | val=0.5318 | dice=0.5208 | best=0.5208 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5108 | val=0.5046 | dice=0.6364 | best=0.6364 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4834 | val=0.4749 | dice=0.6743 | best=0.6743 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4614 | val=0.4631 | dice=0.6808 | best=0.6808 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4474 | val=0.4487 | dice=0.7233 | best=0.7233 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4389 | val=0.4414 | dice=0.7348 | best=0.7348 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4338 | val=0.4392 | dice=0.7405 | best=0.7405 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7405 | VRAM=0MB

&#x20; \[Probe 019/54] → dice=0.7405 | VRAM=0MB



==============================================================

&#x20; \[Probe 020/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6152 | val=0.5714 | dice=0.4654 | best=0.4654 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5420 | val=0.5506 | dice=0.4629 | best=0.4654 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5085 | val=0.5042 | dice=0.6272 | best=0.6272 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4830 | val=0.4750 | dice=0.6484 | best=0.6484 | lr=5.05e-05 | VRAM=487MB | 40s

&#x20; Epoch \[  5/8] train=0.4612 | val=0.4642 | dice=0.6796 | best=0.6796 | lr=3.16e-05 | VRAM=487MB | 50s

&#x20; Epoch \[  6/8] train=0.4468 | val=0.4482 | dice=0.7233 | best=0.7233 | lr=1.55e-05 | VRAM=487MB | 60s

&#x20; Epoch \[  7/8] train=0.4383 | val=0.4429 | dice=0.7280 | best=0.7280 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4335 | val=0.4401 | dice=0.7449 | best=0.7449 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7449 | VRAM=0MB

&#x20; \[Probe 020/54] → dice=0.7449 | VRAM=0MB



==============================================================

&#x20; \[Probe 021/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6155 | val=0.5796 | dice=0.4539 | best=0.4539 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5422 | val=0.5534 | dice=0.4731 | best=0.4731 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5104 | val=0.5050 | dice=0.6334 | best=0.6334 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4829 | val=0.4725 | dice=0.6642 | best=0.6642 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4607 | val=0.4591 | dice=0.6751 | best=0.6751 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4470 | val=0.4497 | dice=0.7193 | best=0.7193 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4387 | val=0.4413 | dice=0.7318 | best=0.7318 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4338 | val=0.4391 | dice=0.7453 | best=0.7453 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7453 | VRAM=0MB

&#x20; \[Probe 021/54] → dice=0.7453 | VRAM=0MB



==============================================================

&#x20; \[Probe 022/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6149 | val=0.5699 | dice=0.4359 | best=0.4359 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5415 | val=0.5631 | dice=0.4332 | best=0.4359 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5080 | val=0.5017 | dice=0.6313 | best=0.6313 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4814 | val=0.4789 | dice=0.6672 | best=0.6672 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4599 | val=0.4560 | dice=0.6915 | best=0.6915 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4459 | val=0.4470 | dice=0.7301 | best=0.7301 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4374 | val=0.4404 | dice=0.7391 | best=0.7391 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4326 | val=0.4392 | dice=0.7484 | best=0.7484 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7484 | VRAM=0MB

&#x20; \[Probe 022/54] → dice=0.7484 | VRAM=0MB



==============================================================

&#x20; \[Probe 023/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5704 | dice=0.4845 | best=0.4845 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5408 | val=0.5265 | dice=0.5631 | best=0.5631 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5089 | val=0.4988 | dice=0.6161 | best=0.6161 | lr=6.94e-05 | VRAM=487MB | 31s

&#x20; Epoch \[  4/8] train=0.4816 | val=0.4991 | dice=0.4457 | best=0.6161 | lr=5.05e-05 | VRAM=487MB | 43s

&#x20; Epoch \[  5/8] train=0.4635 | val=0.4584 | dice=0.6873 | best=0.6873 | lr=3.16e-05 | VRAM=487MB | 54s

&#x20; Epoch \[  6/8] train=0.4478 | val=0.4479 | dice=0.7257 | best=0.7257 | lr=1.55e-05 | VRAM=487MB | 64s

&#x20; Epoch \[  7/8] train=0.4390 | val=0.4427 | dice=0.7334 | best=0.7334 | lr=4.77e-06 | VRAM=487MB | 74s

&#x20; Epoch \[  8/8] train=0.4343 | val=0.4398 | dice=0.7403 | best=0.7403 | lr=1.00e-06 | VRAM=487MB | 84s

→ dice=0.7403 | VRAM=0MB

&#x20; \[Probe 023/54] → dice=0.7403 | VRAM=0MB



==============================================================

&#x20; \[Probe 024/54] lr=0.00019  bs=16

==============================================================



&#x20; \[probe] lr=0.00019, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6007 | val=0.6233 | dice=0.3240 | best=0.3240 | lr=1.79e-04 | VRAM=487MB | 11s

&#x20; Epoch \[  2/8] train=0.5177 | val=0.5104 | dice=0.4872 | best=0.4872 | lr=1.59e-04 | VRAM=487MB | 22s

&#x20; Epoch \[  3/8] train=0.4719 | val=0.4579 | dice=0.6310 | best=0.6310 | lr=1.29e-04 | VRAM=487MB | 34s

&#x20; Epoch \[  4/8] train=0.4313 | val=0.4173 | dice=0.6620 | best=0.6620 | lr=9.37e-05 | VRAM=487MB | 46s

&#x20; Epoch \[  5/8] train=0.4009 | val=0.3957 | dice=0.7001 | best=0.7001 | lr=5.82e-05 | VRAM=487MB | 57s

&#x20; Epoch \[  6/8] train=0.3806 | val=0.3813 | dice=0.7103 | best=0.7103 | lr=2.82e-05 | VRAM=487MB | 69s

&#x20; Epoch \[  7/8] train=0.3673 | val=0.3692 | dice=0.7225 | best=0.7225 | lr=8.06e-06 | VRAM=487MB | 81s

&#x20; Epoch \[  8/8] train=0.3595 | val=0.3659 | dice=0.7342 | best=0.7342 | lr=1.00e-06 | VRAM=487MB | 93s

→ dice=0.7342 | VRAM=0MB

&#x20; \[Probe 024/54] → dice=0.7342 | VRAM=0MB



&#x20; \[Iter  3/8 SUMMARY] w=0.775 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] ── Iter  4/8 (w=0.713) ──────────────────────



==============================================================

&#x20; \[Probe 025/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6147 | val=0.5732 | dice=0.3997 | best=0.3997 | lr=9.62e-05 | VRAM=487MB | 12s

&#x20; Epoch \[  2/8] train=0.5446 | val=0.5670 | dice=0.4144 | best=0.4144 | lr=8.55e-05 | VRAM=487MB | 24s

&#x20; Epoch \[  3/8] train=0.5099 | val=0.5028 | dice=0.6265 | best=0.6265 | lr=6.94e-05 | VRAM=487MB | 35s

&#x20; Epoch \[  4/8] train=0.4832 | val=0.4726 | dice=0.6643 | best=0.6643 | lr=5.05e-05 | VRAM=487MB | 47s

&#x20; Epoch \[  5/8] train=0.4624 | val=0.4595 | dice=0.6868 | best=0.6868 | lr=3.16e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  6/8] train=0.4484 | val=0.4497 | dice=0.7390 | best=0.7390 | lr=1.55e-05 | VRAM=487MB | 71s

&#x20; Epoch \[  7/8] train=0.4395 | val=0.4430 | dice=0.7343 | best=0.7390 | lr=4.77e-06 | VRAM=487MB | 83s

&#x20; Epoch \[  8/8] train=0.4345 | val=0.4400 | dice=0.7429 | best=0.7429 | lr=1.00e-06 | VRAM=487MB | 95s

→ dice=0.7429 | VRAM=0MB

&#x20; \[Probe 025/54] → dice=0.7429 | VRAM=0MB



==============================================================

&#x20; \[Probe 026/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6151 | val=0.5683 | dice=0.4625 | best=0.4625 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5397 | val=0.5341 | dice=0.5293 | best=0.5293 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5084 | val=0.4993 | dice=0.6094 | best=0.6094 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4819 | val=0.4712 | dice=0.6558 | best=0.6558 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4610 | val=0.4531 | dice=0.7019 | best=0.7019 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4474 | val=0.4478 | dice=0.7346 | best=0.7346 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4384 | val=0.4419 | dice=0.7372 | best=0.7372 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4337 | val=0.4383 | dice=0.7437 | best=0.7437 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7437 | VRAM=0MB

&#x20; \[Probe 026/54] → dice=0.7437 | VRAM=0MB



==============================================================

&#x20; \[Probe 027/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6149 | val=0.5734 | dice=0.4227 | best=0.4227 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5410 | val=0.5447 | dice=0.4902 | best=0.4902 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5112 | val=0.5022 | dice=0.6305 | best=0.6305 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4821 | val=0.4731 | dice=0.6713 | best=0.6713 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4609 | val=0.4511 | dice=0.7020 | best=0.7020 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4470 | val=0.4450 | dice=0.7321 | best=0.7321 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4386 | val=0.4409 | dice=0.7323 | best=0.7323 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4337 | val=0.4395 | dice=0.7413 | best=0.7413 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7413 | VRAM=0MB

&#x20; \[Probe 027/54] → dice=0.7413 | VRAM=0MB



==============================================================

&#x20; \[Probe 028/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6158 | val=0.5781 | dice=0.4266 | best=0.4266 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5415 | val=0.5531 | dice=0.4615 | best=0.4615 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5104 | val=0.5089 | dice=0.6379 | best=0.6379 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4826 | val=0.4757 | dice=0.6774 | best=0.6774 | lr=5.05e-05 | VRAM=487MB | 40s

&#x20; Epoch \[  5/8] train=0.4613 | val=0.4615 | dice=0.6939 | best=0.6939 | lr=3.16e-05 | VRAM=487MB | 50s

&#x20; Epoch \[  6/8] train=0.4469 | val=0.4455 | dice=0.7296 | best=0.7296 | lr=1.55e-05 | VRAM=487MB | 60s

&#x20; Epoch \[  7/8] train=0.4382 | val=0.4433 | dice=0.7383 | best=0.7383 | lr=4.77e-06 | VRAM=487MB | 70s

&#x20; Epoch \[  8/8] train=0.4335 | val=0.4383 | dice=0.7479 | best=0.7479 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7479 | VRAM=0MB

&#x20; \[Probe 028/54] → dice=0.7479 | VRAM=0MB



==============================================================

&#x20; \[Probe 029/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6161 | val=0.5796 | dice=0.3886 | best=0.3886 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5404 | val=0.5257 | dice=0.5596 | best=0.5596 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5085 | val=0.4978 | dice=0.6384 | best=0.6384 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4822 | val=0.4744 | dice=0.6758 | best=0.6758 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4601 | val=0.4581 | dice=0.6983 | best=0.6983 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4460 | val=0.4480 | dice=0.7294 | best=0.7294 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4380 | val=0.4411 | dice=0.7349 | best=0.7349 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4332 | val=0.4402 | dice=0.7463 | best=0.7463 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7463 | VRAM=0MB

&#x20; \[Probe 029/54] → dice=0.7463 | VRAM=0MB



==============================================================

&#x20; \[Probe 030/54] lr=0.00020  bs=16

==============================================================



&#x20; \[probe] lr=0.00020, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.5993 | val=0.5843 | dice=0.3599 | best=0.3599 | lr=1.94e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5186 | val=0.5437 | dice=0.4218 | best=0.4218 | lr=1.72e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.4685 | val=0.4572 | dice=0.6167 | best=0.6167 | lr=1.39e-04 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4282 | val=0.4078 | dice=0.6644 | best=0.6644 | lr=1.01e-04 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.3958 | val=0.3930 | dice=0.6713 | best=0.6713 | lr=6.28e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.3737 | val=0.3715 | dice=0.7290 | best=0.7290 | lr=3.03e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.3588 | val=0.3603 | dice=0.7343 | best=0.7343 | lr=8.62e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.3506 | val=0.3568 | dice=0.7411 | best=0.7411 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7411 | VRAM=0MB

&#x20; \[Probe 030/54] → dice=0.7411 | VRAM=0MB



&#x20; \[Iter  4/8 SUMMARY] w=0.713 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] ── Iter  5/8 (w=0.650) ──────────────────────



==============================================================

&#x20; \[Probe 031/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6161 | val=0.5743 | dice=0.4025 | best=0.4025 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5427 | val=0.5495 | dice=0.4584 | best=0.4584 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5128 | val=0.5129 | dice=0.5620 | best=0.5620 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4848 | val=0.4779 | dice=0.6564 | best=0.6564 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4626 | val=0.4611 | dice=0.6801 | best=0.6801 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4487 | val=0.4497 | dice=0.7273 | best=0.7273 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4404 | val=0.4435 | dice=0.7265 | best=0.7273 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4356 | val=0.4417 | dice=0.7368 | best=0.7368 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7368 | VRAM=0MB

&#x20; \[Probe 031/54] → dice=0.7368 | VRAM=0MB



==============================================================

&#x20; \[Probe 032/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6147 | val=0.5688 | dice=0.4532 | best=0.4532 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5417 | val=0.5617 | dice=0.4405 | best=0.4532 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5094 | val=0.5026 | dice=0.6395 | best=0.6395 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4821 | val=0.4770 | dice=0.6721 | best=0.6721 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4613 | val=0.4628 | dice=0.6880 | best=0.6880 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4468 | val=0.4445 | dice=0.7282 | best=0.7282 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4379 | val=0.4413 | dice=0.7337 | best=0.7337 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4333 | val=0.4394 | dice=0.7459 | best=0.7459 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7459 | VRAM=0MB

&#x20; \[Probe 032/54] → dice=0.7459 | VRAM=0MB



==============================================================

&#x20; \[Probe 033/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6159 | val=0.5791 | dice=0.4561 | best=0.4561 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5447 | val=0.5522 | dice=0.4549 | best=0.4561 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5105 | val=0.5037 | dice=0.6079 | best=0.6079 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4850 | val=0.4787 | dice=0.6603 | best=0.6603 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4623 | val=0.4652 | dice=0.6757 | best=0.6757 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4483 | val=0.4479 | dice=0.7179 | best=0.7179 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4395 | val=0.4439 | dice=0.7269 | best=0.7269 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4347 | val=0.4400 | dice=0.7398 | best=0.7398 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7398 | VRAM=0MB

&#x20; \[Probe 033/54] → dice=0.7398 | VRAM=0MB



==============================================================

&#x20; \[Probe 034/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6160 | val=0.5679 | dice=0.4344 | best=0.4344 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5415 | val=0.5455 | dice=0.4700 | best=0.4700 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5101 | val=0.5068 | dice=0.5769 | best=0.5769 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4824 | val=0.4752 | dice=0.6674 | best=0.6674 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4616 | val=0.4584 | dice=0.6819 | best=0.6819 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4475 | val=0.4480 | dice=0.7343 | best=0.7343 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4384 | val=0.4423 | dice=0.7383 | best=0.7383 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4335 | val=0.4392 | dice=0.7469 | best=0.7469 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7469 | VRAM=0MB

&#x20; \[Probe 034/54] → dice=0.7469 | VRAM=0MB



==============================================================

&#x20; \[Probe 035/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6147 | val=0.5753 | dice=0.4252 | best=0.4252 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5510 | val=0.5861 | dice=0.4005 | best=0.4252 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5125 | val=0.5034 | dice=0.6299 | best=0.6299 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4841 | val=0.4787 | dice=0.6450 | best=0.6450 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4634 | val=0.4624 | dice=0.6856 | best=0.6856 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4488 | val=0.4485 | dice=0.7260 | best=0.7260 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4400 | val=0.4428 | dice=0.7276 | best=0.7276 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4350 | val=0.4404 | dice=0.7427 | best=0.7427 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7427 | VRAM=0MB

&#x20; \[Probe 035/54] → dice=0.7427 | VRAM=0MB



==============================================================

&#x20; \[Probe 036/54] lr=0.00015  bs=16

==============================================================



&#x20; \[probe] lr=0.00015, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6056 | val=0.5724 | dice=0.4351 | best=0.4351 | lr=1.43e-04 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5257 | val=0.5069 | dice=0.5206 | best=0.5206 | lr=1.27e-04 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.4852 | val=0.4778 | dice=0.6065 | best=0.6065 | lr=1.03e-04 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4509 | val=0.4362 | dice=0.6512 | best=0.6512 | lr=7.47e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4247 | val=0.4256 | dice=0.6973 | best=0.6973 | lr=4.65e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4072 | val=0.4089 | dice=0.7198 | best=0.7198 | lr=2.26e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.3956 | val=0.3966 | dice=0.7349 | best=0.7349 | lr=6.61e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.3893 | val=0.3959 | dice=0.7407 | best=0.7407 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7407 | VRAM=0MB

&#x20; \[Probe 036/54] → dice=0.7407 | VRAM=0MB



&#x20; \[Iter  5/8 SUMMARY] w=0.650 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] ── Iter  6/8 (w=0.588) ──────────────────────



==============================================================

&#x20; \[Probe 037/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5803 | dice=0.4251 | best=0.4251 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5418 | val=0.5278 | dice=0.5362 | best=0.5362 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5093 | val=0.5245 | dice=0.5325 | best=0.5362 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4843 | val=0.4716 | dice=0.6662 | best=0.6662 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4604 | val=0.4591 | dice=0.6875 | best=0.6875 | lr=3.16e-05 | VRAM=487MB | 48s

&#x20; Epoch \[  6/8] train=0.4469 | val=0.4491 | dice=0.7222 | best=0.7222 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4385 | val=0.4430 | dice=0.7377 | best=0.7377 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4336 | val=0.4391 | dice=0.7438 | best=0.7438 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7438 | VRAM=0MB

&#x20; \[Probe 037/54] → dice=0.7438 | VRAM=0MB



==============================================================

&#x20; \[Probe 038/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6154 | val=0.5738 | dice=0.3769 | best=0.3769 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5426 | val=0.5619 | dice=0.4489 | best=0.4489 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5097 | val=0.5065 | dice=0.6268 | best=0.6268 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4831 | val=0.4777 | dice=0.6868 | best=0.6868 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4613 | val=0.4574 | dice=0.6795 | best=0.6868 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4475 | val=0.4473 | dice=0.7205 | best=0.7205 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4386 | val=0.4415 | dice=0.7317 | best=0.7317 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4338 | val=0.4389 | dice=0.7413 | best=0.7413 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7413 | VRAM=0MB

&#x20; \[Probe 038/54] → dice=0.7413 | VRAM=0MB



==============================================================

&#x20; \[Probe 039/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6153 | val=0.5708 | dice=0.4003 | best=0.4003 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5419 | val=0.5504 | dice=0.4664 | best=0.4664 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5096 | val=0.5011 | dice=0.6426 | best=0.6426 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4820 | val=0.4781 | dice=0.6637 | best=0.6637 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4615 | val=0.4566 | dice=0.6879 | best=0.6879 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4478 | val=0.4482 | dice=0.7266 | best=0.7266 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4388 | val=0.4426 | dice=0.7281 | best=0.7281 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4341 | val=0.4401 | dice=0.7421 | best=0.7421 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7421 | VRAM=0MB

&#x20; \[Probe 039/54] → dice=0.7421 | VRAM=0MB



==============================================================

&#x20; \[Probe 040/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5828 | dice=0.4574 | best=0.4574 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5409 | val=0.5757 | dice=0.3978 | best=0.4574 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5114 | val=0.5103 | dice=0.6279 | best=0.6279 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4853 | val=0.4749 | dice=0.6525 | best=0.6525 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4622 | val=0.4614 | dice=0.6751 | best=0.6751 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4482 | val=0.4480 | dice=0.7207 | best=0.7207 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4399 | val=0.4431 | dice=0.7261 | best=0.7261 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4353 | val=0.4406 | dice=0.7429 | best=0.7429 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7429 | VRAM=0MB

&#x20; \[Probe 040/54] → dice=0.7429 | VRAM=0MB



==============================================================

&#x20; \[Probe 041/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6148 | val=0.5670 | dice=0.4586 | best=0.4586 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5398 | val=0.5459 | dice=0.4826 | best=0.4826 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5085 | val=0.4994 | dice=0.6392 | best=0.6392 | lr=6.94e-05 | VRAM=487MB | 30s

&#x20; Epoch \[  4/8] train=0.4811 | val=0.4803 | dice=0.6615 | best=0.6615 | lr=5.05e-05 | VRAM=487MB | 40s

&#x20; Epoch \[  5/8] train=0.4609 | val=0.4589 | dice=0.6882 | best=0.6882 | lr=3.16e-05 | VRAM=487MB | 50s

&#x20; Epoch \[  6/8] train=0.4463 | val=0.4493 | dice=0.7248 | best=0.7248 | lr=1.55e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  7/8] train=0.4381 | val=0.4400 | dice=0.7308 | best=0.7308 | lr=4.77e-06 | VRAM=487MB | 69s

&#x20; Epoch \[  8/8] train=0.4333 | val=0.4389 | dice=0.7395 | best=0.7395 | lr=1.00e-06 | VRAM=487MB | 79s

→ dice=0.7395 | VRAM=0MB

&#x20; \[Probe 041/54] → dice=0.7395 | VRAM=0MB



==============================================================

&#x20; \[Probe 042/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6153 | val=0.5680 | dice=0.4235 | best=0.4235 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5410 | val=0.5466 | dice=0.4727 | best=0.4727 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5094 | val=0.5029 | dice=0.6354 | best=0.6354 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4817 | val=0.4751 | dice=0.6779 | best=0.6779 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4603 | val=0.4605 | dice=0.6842 | best=0.6842 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4470 | val=0.4464 | dice=0.7282 | best=0.7282 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4382 | val=0.4420 | dice=0.7376 | best=0.7376 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4334 | val=0.4396 | dice=0.7426 | best=0.7426 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7426 | VRAM=0MB

&#x20; \[Probe 042/54] → dice=0.7426 | VRAM=0MB



&#x20; \[Iter  6/8 SUMMARY] w=0.588 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] ── Iter  7/8 (w=0.525) ──────────────────────



==============================================================

&#x20; \[Probe 043/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5756 | dice=0.4079 | best=0.4079 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5411 | val=0.5311 | dice=0.5506 | best=0.5506 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5088 | val=0.5005 | dice=0.6116 | best=0.6116 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4814 | val=0.4724 | dice=0.6884 | best=0.6884 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4605 | val=0.4521 | dice=0.6720 | best=0.6884 | lr=3.16e-05 | VRAM=487MB | 48s

&#x20; Epoch \[  6/8] train=0.4466 | val=0.4482 | dice=0.7210 | best=0.7210 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4380 | val=0.4409 | dice=0.7265 | best=0.7265 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4334 | val=0.4392 | dice=0.7366 | best=0.7366 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7366 | VRAM=0MB

&#x20; \[Probe 043/54] → dice=0.7366 | VRAM=0MB



==============================================================

&#x20; \[Probe 044/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6148 | val=0.5668 | dice=0.4528 | best=0.4528 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5400 | val=0.5313 | dice=0.5454 | best=0.5454 | lr=8.55e-05 | VRAM=487MB | 20s

&#x20; Epoch \[  3/8] train=0.5097 | val=0.4974 | dice=0.6442 | best=0.6442 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4819 | val=0.4747 | dice=0.6478 | best=0.6478 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4602 | val=0.4586 | dice=0.6830 | best=0.6830 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4469 | val=0.4442 | dice=0.7246 | best=0.7246 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4381 | val=0.4426 | dice=0.7294 | best=0.7294 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4337 | val=0.4391 | dice=0.7396 | best=0.7396 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7396 | VRAM=0MB

&#x20; \[Probe 044/54] → dice=0.7396 | VRAM=0MB



==============================================================

&#x20; \[Probe 045/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6160 | val=0.5720 | dice=0.4465 | best=0.4465 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5403 | val=0.5311 | dice=0.5232 | best=0.5232 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5099 | val=0.4984 | dice=0.6313 | best=0.6313 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4821 | val=0.4770 | dice=0.6874 | best=0.6874 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4606 | val=0.4590 | dice=0.6962 | best=0.6962 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4469 | val=0.4516 | dice=0.7346 | best=0.7346 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4381 | val=0.4427 | dice=0.7303 | best=0.7346 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4334 | val=0.4396 | dice=0.7439 | best=0.7439 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7439 | VRAM=0MB

&#x20; \[Probe 045/54] → dice=0.7439 | VRAM=0MB



==============================================================

&#x20; \[Probe 046/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6149 | val=0.5736 | dice=0.4134 | best=0.4134 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5440 | val=0.5494 | dice=0.4685 | best=0.4685 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5103 | val=0.4981 | dice=0.6227 | best=0.6227 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4856 | val=0.4784 | dice=0.6607 | best=0.6607 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4624 | val=0.4537 | dice=0.6948 | best=0.6948 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4478 | val=0.4500 | dice=0.7153 | best=0.7153 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4395 | val=0.4424 | dice=0.7244 | best=0.7244 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4348 | val=0.4399 | dice=0.7412 | best=0.7412 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7412 | VRAM=0MB

&#x20; \[Probe 046/54] → dice=0.7412 | VRAM=0MB



==============================================================

&#x20; \[Probe 047/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6150 | val=0.5652 | dice=0.4311 | best=0.4311 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5427 | val=0.5540 | dice=0.4480 | best=0.4480 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5098 | val=0.5016 | dice=0.6207 | best=0.6207 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4830 | val=0.4829 | dice=0.6386 | best=0.6386 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4616 | val=0.4595 | dice=0.6952 | best=0.6952 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4479 | val=0.4485 | dice=0.7173 | best=0.7173 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4389 | val=0.4427 | dice=0.7334 | best=0.7334 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4340 | val=0.4392 | dice=0.7388 | best=0.7388 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7388 | VRAM=0MB

&#x20; \[Probe 047/54] → dice=0.7388 | VRAM=0MB



==============================================================

&#x20; \[Probe 048/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6169 | val=0.5745 | dice=0.4510 | best=0.4510 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5419 | val=0.5389 | dice=0.4945 | best=0.4945 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5094 | val=0.5016 | dice=0.6129 | best=0.6129 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4829 | val=0.4778 | dice=0.6535 | best=0.6535 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4616 | val=0.4642 | dice=0.6688 | best=0.6688 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4477 | val=0.4505 | dice=0.7355 | best=0.7355 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4397 | val=0.4415 | dice=0.7360 | best=0.7360 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4343 | val=0.4402 | dice=0.7439 | best=0.7439 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7439 | VRAM=0MB

&#x20; \[Probe 048/54] → dice=0.7439 | VRAM=0MB



&#x20; \[Iter  7/8 SUMMARY] w=0.525 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] ── Iter  8/8 (w=0.463) ──────────────────────



==============================================================

&#x20; \[Probe 049/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5739 | dice=0.4229 | best=0.4229 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5454 | val=0.5880 | dice=0.3922 | best=0.4229 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5162 | val=0.5023 | dice=0.6412 | best=0.6412 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4861 | val=0.4805 | dice=0.6417 | best=0.6417 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4646 | val=0.4603 | dice=0.7024 | best=0.7024 | lr=3.16e-05 | VRAM=487MB | 48s

&#x20; Epoch \[  6/8] train=0.4509 | val=0.4492 | dice=0.7277 | best=0.7277 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4416 | val=0.4426 | dice=0.7219 | best=0.7277 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4371 | val=0.4416 | dice=0.7342 | best=0.7342 | lr=1.00e-06 | VRAM=487MB | 77s

→ dice=0.7342 | VRAM=0MB

&#x20; \[Probe 049/54] → dice=0.7342 | VRAM=0MB



==============================================================

&#x20; \[Probe 050/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6151 | val=0.5731 | dice=0.4164 | best=0.4164 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5413 | val=0.5630 | dice=0.4330 | best=0.4330 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5093 | val=0.4975 | dice=0.6432 | best=0.6432 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4814 | val=0.4727 | dice=0.6867 | best=0.6867 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4598 | val=0.4572 | dice=0.6934 | best=0.6934 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4464 | val=0.4461 | dice=0.7247 | best=0.7247 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4377 | val=0.4417 | dice=0.7275 | best=0.7275 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4330 | val=0.4389 | dice=0.7449 | best=0.7449 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7449 | VRAM=0MB

&#x20; \[Probe 050/54] → dice=0.7449 | VRAM=0MB



==============================================================

&#x20; \[Probe 051/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6152 | val=0.5818 | dice=0.3518 | best=0.3518 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5427 | val=0.5401 | dice=0.5165 | best=0.5165 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5091 | val=0.4992 | dice=0.6306 | best=0.6306 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4824 | val=0.4788 | dice=0.6635 | best=0.6635 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4611 | val=0.4579 | dice=0.6838 | best=0.6838 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4468 | val=0.4482 | dice=0.7242 | best=0.7242 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4381 | val=0.4423 | dice=0.7367 | best=0.7367 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4334 | val=0.4385 | dice=0.7429 | best=0.7429 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7429 | VRAM=0MB

&#x20; \[Probe 051/54] → dice=0.7429 | VRAM=0MB



==============================================================

&#x20; \[Probe 052/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6157 | val=0.5721 | dice=0.4426 | best=0.4426 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5417 | val=0.5473 | dice=0.4555 | best=0.4555 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5092 | val=0.4983 | dice=0.6423 | best=0.6423 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4826 | val=0.4796 | dice=0.6606 | best=0.6606 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4606 | val=0.4587 | dice=0.6927 | best=0.6927 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4469 | val=0.4452 | dice=0.7270 | best=0.7270 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4381 | val=0.4426 | dice=0.7342 | best=0.7342 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4334 | val=0.4390 | dice=0.7434 | best=0.7434 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7434 | VRAM=0MB

&#x20; \[Probe 052/54] → dice=0.7434 | VRAM=0MB



==============================================================

&#x20; \[Probe 053/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6150 | val=0.5726 | dice=0.4714 | best=0.4714 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5419 | val=0.5547 | dice=0.4396 | best=0.4714 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5104 | val=0.5160 | dice=0.5737 | best=0.5737 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4848 | val=0.4747 | dice=0.6670 | best=0.6670 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4621 | val=0.4570 | dice=0.7026 | best=0.7026 | lr=3.16e-05 | VRAM=487MB | 48s

&#x20; Epoch \[  6/8] train=0.4479 | val=0.4473 | dice=0.7268 | best=0.7268 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4396 | val=0.4419 | dice=0.7371 | best=0.7371 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4350 | val=0.4401 | dice=0.7451 | best=0.7451 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7451 | VRAM=0MB

&#x20; \[Probe 053/54] → dice=0.7451 | VRAM=0MB



==============================================================

&#x20; \[Probe 054/54] lr=0.00010  bs=16

==============================================================



&#x20; \[probe] lr=0.00010, bs=16 \[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/8] train=0.6156 | val=0.5723 | dice=0.4306 | best=0.4306 | lr=9.62e-05 | VRAM=487MB | 10s

&#x20; Epoch \[  2/8] train=0.5406 | val=0.5339 | dice=0.5211 | best=0.5211 | lr=8.55e-05 | VRAM=487MB | 19s

&#x20; Epoch \[  3/8] train=0.5134 | val=0.5181 | dice=0.6097 | best=0.6097 | lr=6.94e-05 | VRAM=487MB | 29s

&#x20; Epoch \[  4/8] train=0.4853 | val=0.4784 | dice=0.6650 | best=0.6650 | lr=5.05e-05 | VRAM=487MB | 39s

&#x20; Epoch \[  5/8] train=0.4625 | val=0.4571 | dice=0.7023 | best=0.7023 | lr=3.16e-05 | VRAM=487MB | 49s

&#x20; Epoch \[  6/8] train=0.4485 | val=0.4502 | dice=0.7217 | best=0.7217 | lr=1.55e-05 | VRAM=487MB | 58s

&#x20; Epoch \[  7/8] train=0.4400 | val=0.4421 | dice=0.7229 | best=0.7229 | lr=4.77e-06 | VRAM=487MB | 68s

&#x20; Epoch \[  8/8] train=0.4352 | val=0.4416 | dice=0.7369 | best=0.7369 | lr=1.00e-06 | VRAM=487MB | 78s

→ dice=0.7369 | VRAM=0MB

&#x20; \[Probe 054/54] → dice=0.7369 | VRAM=0MB



&#x20; \[Iter  8/8 SUMMARY] w=0.463 | best\_dice=0.7484 | lr=0.00010, bs=16



\[PSO-Hyper] BEST → {'lr': 0.0001, 'batch\_size': 16} | Dice=0.7484





\[PSO] Best hyperparams : {'lr': 0.0001, 'batch\_size': 16}

\[PSO] Best probe Dice  : 0.7484



\[Train] Retraining for 50 epochs ...

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Epoch \[  1/50] train=0.6172 | val=0.5844 | dice=0.4011 | best=0.4011 | lr=9.99e-05 | VRAM=487MB | 34s

&#x20; Epoch \[  2/50] train=0.5426 | val=0.5265 | dice=0.5553 | best=0.5553 | lr=9.96e-05 | VRAM=487MB | 42s

&#x20; Epoch \[  3/50] train=0.5082 | val=0.4986 | dice=0.6127 | best=0.6127 | lr=9.91e-05 | VRAM=487MB | 51s

&#x20; Epoch \[  4/50] train=0.4795 | val=0.4641 | dice=0.6466 | best=0.6466 | lr=9.84e-05 | VRAM=487MB | 59s

&#x20; Epoch \[  5/50] train=0.4505 | val=0.4395 | dice=0.6869 | best=0.6869 | lr=9.76e-05 | VRAM=487MB | 68s

&#x20; Epoch \[  6/50] train=0.4259 | val=0.4075 | dice=0.7007 | best=0.7007 | lr=9.65e-05 | VRAM=487MB | 76s

&#x20; Epoch \[  7/50] train=0.4038 | val=0.4103 | dice=0.7184 | best=0.7184 | lr=9.53e-05 | VRAM=487MB | 85s

&#x20; Epoch \[  8/50] train=0.3824 | val=0.3802 | dice=0.7226 | best=0.7226 | lr=9.39e-05 | VRAM=487MB | 93s

&#x20; Epoch \[  9/50] train=0.3630 | val=0.3588 | dice=0.7070 | best=0.7226 | lr=9.23e-05 | VRAM=487MB | 102s

&#x20; Epoch \[ 10/50] train=0.3412 | val=0.3405 | dice=0.7169 | best=0.7226 | lr=9.05e-05 | VRAM=487MB | 110s

&#x20; Epoch \[ 11/50] train=0.3217 | val=0.3368 | dice=0.7415 | best=0.7415 | lr=8.86e-05 | VRAM=487MB | 119s

&#x20; Epoch \[ 12/50] train=0.3074 | val=0.3300 | dice=0.7331 | best=0.7415 | lr=8.66e-05 | VRAM=487MB | 127s

&#x20; Epoch \[ 13/50] train=0.2903 | val=0.2961 | dice=0.7429 | best=0.7429 | lr=8.44e-05 | VRAM=487MB | 136s

&#x20; Epoch \[ 14/50] train=0.2754 | val=0.2748 | dice=0.7530 | best=0.7530 | lr=8.21e-05 | VRAM=487MB | 144s

&#x20; Epoch \[ 15/50] train=0.2624 | val=0.2698 | dice=0.7449 | best=0.7530 | lr=7.96e-05 | VRAM=487MB | 153s

&#x20; Epoch \[ 16/50] train=0.2476 | val=0.2648 | dice=0.7684 | best=0.7684 | lr=7.70e-05 | VRAM=487MB | 161s

&#x20; Epoch \[ 17/50] train=0.2373 | val=0.2462 | dice=0.7654 | best=0.7684 | lr=7.43e-05 | VRAM=487MB | 169s

&#x20; Epoch \[ 18/50] train=0.2197 | val=0.2295 | dice=0.8047 | best=0.8047 | lr=7.16e-05 | VRAM=487MB | 178s

&#x20; Epoch \[ 19/50] train=0.2102 | val=0.2203 | dice=0.7897 | best=0.8047 | lr=6.87e-05 | VRAM=487MB | 186s

&#x20; Epoch \[ 20/50] train=0.1980 | val=0.2193 | dice=0.7872 | best=0.8047 | lr=6.58e-05 | VRAM=487MB | 195s

&#x20; Epoch \[ 21/50] train=0.1895 | val=0.2012 | dice=0.7954 | best=0.8047 | lr=6.28e-05 | VRAM=487MB | 203s

&#x20; Epoch \[ 22/50] train=0.1789 | val=0.2016 | dice=0.7919 | best=0.8047 | lr=5.98e-05 | VRAM=487MB | 212s

&#x20; Epoch \[ 23/50] train=0.1683 | val=0.1906 | dice=0.8135 | best=0.8135 | lr=5.67e-05 | VRAM=487MB | 220s

&#x20; Epoch \[ 24/50] train=0.1600 | val=0.1816 | dice=0.8153 | best=0.8153 | lr=5.36e-05 | VRAM=487MB | 229s

&#x20; Epoch \[ 25/50] train=0.1532 | val=0.1719 | dice=0.7953 | best=0.8153 | lr=5.05e-05 | VRAM=487MB | 237s

&#x20; Epoch \[ 26/50] train=0.1522 | val=0.1780 | dice=0.7849 | best=0.8153 | lr=4.74e-05 | VRAM=487MB | 245s

&#x20; Epoch \[ 27/50] train=0.1426 | val=0.1661 | dice=0.8111 | best=0.8153 | lr=4.43e-05 | VRAM=487MB | 254s

&#x20; Epoch \[ 28/50] train=0.1393 | val=0.1643 | dice=0.8123 | best=0.8153 | lr=4.12e-05 | VRAM=487MB | 262s

&#x20; Epoch \[ 29/50] train=0.1337 | val=0.1573 | dice=0.8164 | best=0.8164 | lr=3.82e-05 | VRAM=487MB | 271s

&#x20; Epoch \[ 30/50] train=0.1265 | val=0.1580 | dice=0.8280 | best=0.8280 | lr=3.52e-05 | VRAM=487MB | 279s

&#x20; Epoch \[ 31/50] train=0.1206 | val=0.1496 | dice=0.8185 | best=0.8280 | lr=3.23e-05 | VRAM=487MB | 287s

&#x20; Epoch \[ 32/50] train=0.1168 | val=0.1452 | dice=0.8234 | best=0.8280 | lr=2.94e-05 | VRAM=487MB | 296s

&#x20; Epoch \[ 33/50] train=0.1137 | val=0.1485 | dice=0.8273 | best=0.8280 | lr=2.67e-05 | VRAM=487MB | 304s

&#x20; Epoch \[ 34/50] train=0.1114 | val=0.1525 | dice=0.8294 | best=0.8294 | lr=2.40e-05 | VRAM=487MB | 313s

&#x20; Epoch \[ 35/50] train=0.1068 | val=0.1417 | dice=0.8347 | best=0.8347 | lr=2.14e-05 | VRAM=487MB | 321s

&#x20; Epoch \[ 36/50] train=0.1038 | val=0.1429 | dice=0.8332 | best=0.8347 | lr=1.89e-05 | VRAM=487MB | 330s

&#x20; Epoch \[ 37/50] train=0.1013 | val=0.1410 | dice=0.8327 | best=0.8347 | lr=1.66e-05 | VRAM=487MB | 338s

&#x20; Epoch \[ 38/50] train=0.0983 | val=0.1414 | dice=0.8333 | best=0.8347 | lr=1.44e-05 | VRAM=487MB | 346s

&#x20; Epoch \[ 39/50] train=0.0956 | val=0.1385 | dice=0.8345 | best=0.8347 | lr=1.24e-05 | VRAM=487MB | 355s

&#x20; Epoch \[ 40/50] train=0.0947 | val=0.1357 | dice=0.8400 | best=0.8400 | lr=1.05e-05 | VRAM=487MB | 363s

&#x20; Epoch \[ 41/50] train=0.0922 | val=0.1354 | dice=0.8361 | best=0.8400 | lr=8.71e-06 | VRAM=487MB | 372s

&#x20; Epoch \[ 42/50] train=0.0913 | val=0.1337 | dice=0.8407 | best=0.8407 | lr=7.12e-06 | VRAM=487MB | 380s

&#x20; Epoch \[ 43/50] train=0.0895 | val=0.1341 | dice=0.8377 | best=0.8407 | lr=5.71e-06 | VRAM=487MB | 389s

&#x20; Epoch \[ 44/50] train=0.0887 | val=0.1320 | dice=0.8386 | best=0.8407 | lr=4.48e-06 | VRAM=487MB | 397s

&#x20; Epoch \[ 45/50] train=0.0873 | val=0.1291 | dice=0.8375 | best=0.8407 | lr=3.42e-06 | VRAM=487MB | 405s

&#x20; Epoch \[ 46/50] train=0.0867 | val=0.1323 | dice=0.8378 | best=0.8407 | lr=2.56e-06 | VRAM=487MB | 414s

&#x20; Epoch \[ 47/50] train=0.0863 | val=0.1328 | dice=0.8415 | best=0.8415 | lr=1.88e-06 | VRAM=487MB | 422s

&#x20; Epoch \[ 48/50] train=0.0864 | val=0.1307 | dice=0.8393 | best=0.8415 | lr=1.39e-06 | VRAM=487MB | 431s

&#x20; Epoch \[ 49/50] train=0.0854 | val=0.1310 | dice=0.8399 | best=0.8415 | lr=1.10e-06 | VRAM=487MB | 439s

&#x20; Epoch \[ 50/50] train=0.0851 | val=0.1316 | dice=0.8406 | best=0.8415 | lr=1.00e-06 | VRAM=487MB | 448s



\[PSO Train] Best Dice = 0.8415



\[PSO-Thresh] Optimizing threshold ...

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0



\[PSO-Thresh] 20 particles × 30 iters

\[PSO-Thresh] Initializing ...

\[PSO-Thresh] Init done. Best threshold so far: 0.4456 → dice=0.8415

&#x20; \[Iter  1/30] best\_threshold=0.4383 | dice=0.8419

&#x20; \[Iter  5/30] best\_threshold=0.4385 | dice=0.8419

&#x20; \[Iter 10/30] best\_threshold=0.4385 | dice=0.8419

&#x20; \[Iter 15/30] best\_threshold=0.4385 | dice=0.8419

&#x20; \[Iter 20/30] best\_threshold=0.4385 | dice=0.8419

&#x20; \[Iter 25/30] best\_threshold=0.4378 | dice=0.8419

&#x20; \[Iter 30/30] best\_threshold=0.4385 | dice=0.8419



\[PSO-Thresh] BEST threshold=0.4385 | Dice=0.8419









**Evaluate**

(segmentation) D:\\Brain\_Segmentation>python evaluate.py --data\_dir processed\_dataset --standard\_model best\_model.pth --pso\_model best\_pso\_model.pth --threshold 0.5



\[Device] cuda

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0



\[Eval] Running Otsu baseline ...

&#x20; Dice=0.1531 | IoU=0.0865 | Acc=0.6968



\[Eval] Standard U-Net (best\_model.pth, threshold=0.50) ...

\[Eval] Running Otsu baseline ...

&#x20; Dice=0.1531 | IoU=0.0865 | Acc=0.6968



\[Eval] Standard U-Net (best\_model.pth, threshold=0.50) ...

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

\[Eval] Standard U-Net (best\_model.pth, threshold=0.50) ...

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Dice=0.8168 | IoU=0.7235 | Acc=0.9930

\[Viz] Saved → standard\_unet\_predictions.png



\[Eval] U-Net + PSO (best\_pso\_model.pth, threshold=0.5000) ...

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20; Dice=0.8395 | IoU=0.7486 | Acc=0.9938

\[Viz] Saved → pso\_unet\_predictions.png



=================================================================

=================================================================

=================================================================

=================================================================

Method                             Dice      IoU   Accuracy

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

=================================================================

Method                             Dice      IoU   Accuracy

=================================================================

Otsu Thresholding                0.1531   0.0865     0.6968

Standard U-Net                   0.8168   0.7235     0.9930

U-Net + PSO (Proposed)           0.8395   0.7486     0.9938

=================================================================



\[Plot] Saved → comparison\_chart.png















**Random Search:**

(segmentation) D:\\Brain\_Segmentation>python random\_search.py --data\_dir processed\_dataset --trials 54 --epochs 8



\[Device] NVIDIA GeForce RTX 4060 Laptop GPU | VRAM: 8.0 GB



\[RandomSearch] 54 trials × 8 proxy epochs

\[RandomSearch] lr ∈ \[1e-04, 1e-02] | batch ∈ \[8, 16]

==============================================================



\[Trial 001/54] lr=0.00381  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2486

&#x20;   Epoch \[ 2/8] dice=0.4203

&#x20;   Epoch \[ 3/8] dice=0.4654

&#x20;   Epoch \[ 4/8] dice=0.3944

&#x20;   Epoch \[ 5/8] dice=0.4435

&#x20;   Epoch \[ 6/8] dice=0.5922

&#x20;   Epoch \[ 7/8] dice=0.6205

&#x20;   Epoch \[ 8/8] dice=0.6600

&#x20; → Trial 001 DONE | dice=0.6600 | 91s

&#x20; ★ New best! dice=0.6600



\[Trial 002/54] lr=0.00192  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3105

&#x20;   Epoch \[ 2/8] dice=0.3501

&#x20;   Epoch \[ 3/8] dice=0.4821

&#x20;   Epoch \[ 4/8] dice=0.4551

&#x20;   Epoch \[ 5/8] dice=0.4757

&#x20;   Epoch \[ 6/8] dice=0.5482

&#x20;   Epoch \[ 7/8] dice=0.5234

&#x20;   Epoch \[ 8/8] dice=0.5906

&#x20; → Trial 002 DONE | dice=0.5906 | 88s



\[Trial 003/54] lr=0.00603  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4562

&#x20;   Epoch \[ 2/8] dice=0.2395

&#x20;   Epoch \[ 3/8] dice=0.4377

&#x20;   Epoch \[ 4/8] dice=0.5413

&#x20;   Epoch \[ 5/8] dice=0.4457

&#x20;   Epoch \[ 6/8] dice=0.3143

&#x20;   Epoch \[ 7/8] dice=0.5678

&#x20;   Epoch \[ 8/8] dice=0.6813

&#x20; → Trial 003 DONE | dice=0.6813 | 92s

&#x20; ★ New best! dice=0.6813



\[Trial 004/54] lr=0.00451  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3332

&#x20;   Epoch \[ 2/8] dice=0.3369

&#x20;   Epoch \[ 3/8] dice=0.2591

&#x20;   Epoch \[ 4/8] dice=0.4904

&#x20;   Epoch \[ 5/8] dice=0.5382

&#x20;   Epoch \[ 6/8] dice=0.6160

&#x20;   Epoch \[ 7/8] dice=0.6486

&#x20;   Epoch \[ 8/8] dice=0.6671

&#x20; → Trial 004 DONE | dice=0.6671 | 75s



\[Trial 005/54] lr=0.00068  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2942

&#x20;   Epoch \[ 2/8] dice=0.4042

&#x20;   Epoch \[ 3/8] dice=0.4450

&#x20;   Epoch \[ 4/8] dice=0.4825

&#x20;   Epoch \[ 5/8] dice=0.5617

&#x20;   Epoch \[ 6/8] dice=0.6099

&#x20;   Epoch \[ 7/8] dice=0.6793

&#x20;   Epoch \[ 8/8] dice=0.7126

&#x20; → Trial 005 DONE | dice=0.7126 | 71s

&#x20; ★ New best! dice=0.7126



\[Trial 006/54] lr=0.00340  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3449

&#x20;   Epoch \[ 2/8] dice=0.4627

&#x20;   Epoch \[ 3/8] dice=0.5082

&#x20;   Epoch \[ 4/8] dice=0.3269

&#x20;   Epoch \[ 5/8] dice=0.3335

&#x20;   Epoch \[ 6/8] dice=0.5351

&#x20;   Epoch \[ 7/8] dice=0.5608

&#x20;   Epoch \[ 8/8] dice=0.5835

&#x20; → Trial 006 DONE | dice=0.5835 | 71s



\[Trial 007/54] lr=0.00711  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3988

&#x20;   Epoch \[ 2/8] dice=0.3269

&#x20;   Epoch \[ 3/8] dice=0.3132

&#x20;   Epoch \[ 4/8] dice=0.3615

&#x20;   Epoch \[ 5/8] dice=0.5203

&#x20;   Epoch \[ 6/8] dice=0.5260

&#x20;   Epoch \[ 7/8] dice=0.6153

&#x20;   Epoch \[ 8/8] dice=0.6553

&#x20; → Trial 007 DONE | dice=0.6553 | 71s



\[Trial 008/54] lr=0.00066  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4146

&#x20;   Epoch \[ 2/8] dice=0.4646

&#x20;   Epoch \[ 3/8] dice=0.5444

&#x20;   Epoch \[ 4/8] dice=0.5929

&#x20;   Epoch \[ 5/8] dice=0.6020

&#x20;   Epoch \[ 6/8] dice=0.6302

&#x20;   Epoch \[ 7/8] dice=0.7024

&#x20;   Epoch \[ 8/8] dice=0.7066

&#x20; → Trial 008 DONE | dice=0.7066 | 71s



\[Trial 009/54] lr=0.00834  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2598

&#x20;   Epoch \[ 2/8] dice=0.0000

&#x20;   Epoch \[ 3/8] dice=0.4871

&#x20;   Epoch \[ 4/8] dice=0.1019

&#x20;   Epoch \[ 5/8] dice=0.5514

&#x20;   Epoch \[ 6/8] dice=0.5407

&#x20;   Epoch \[ 7/8] dice=0.6192

&#x20;   Epoch \[ 8/8] dice=0.6510

&#x20; → Trial 009 DONE | dice=0.6510 | 71s



\[Trial 010/54] lr=0.00011  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3636

&#x20;   Epoch \[ 2/8] dice=0.5424

&#x20;   Epoch \[ 3/8] dice=0.5983

&#x20;   Epoch \[ 4/8] dice=0.6229

&#x20;   Epoch \[ 5/8] dice=0.6771

&#x20;   Epoch \[ 6/8] dice=0.6864

&#x20;   Epoch \[ 7/8] dice=0.7167

&#x20;   Epoch \[ 8/8] dice=0.7204

&#x20; → Trial 010 DONE | dice=0.7204 | 71s

&#x20; ★ New best! dice=0.7204



\[Trial 011/54] lr=0.00192  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4108

&#x20;   Epoch \[ 2/8] dice=0.3921

&#x20;   Epoch \[ 3/8] dice=0.4950

&#x20;   Epoch \[ 4/8] dice=0.5379

&#x20;   Epoch \[ 5/8] dice=0.6100

&#x20;   Epoch \[ 6/8] dice=0.6275

&#x20;   Epoch \[ 7/8] dice=0.6593

&#x20;   Epoch \[ 8/8] dice=0.6851

&#x20; → Trial 011 DONE | dice=0.6851 | 71s



\[Trial 012/54] lr=0.00616  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1729

&#x20;   Epoch \[ 2/8] dice=0.1691

&#x20;   Epoch \[ 3/8] dice=0.4299

&#x20;   Epoch \[ 4/8] dice=0.4424

&#x20;   Epoch \[ 5/8] dice=0.4572

&#x20;   Epoch \[ 6/8] dice=0.6080

&#x20;   Epoch \[ 7/8] dice=0.6071

&#x20;   Epoch \[ 8/8] dice=0.6451

&#x20; → Trial 012 DONE | dice=0.6451 | 75s



\[Trial 013/54] lr=0.00438  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3782

&#x20;   Epoch \[ 2/8] dice=0.4400

&#x20;   Epoch \[ 3/8] dice=0.4669

&#x20;   Epoch \[ 4/8] dice=0.3965

&#x20;   Epoch \[ 5/8] dice=0.5034

&#x20;   Epoch \[ 6/8] dice=0.5430

&#x20;   Epoch \[ 7/8] dice=0.5877

&#x20;   Epoch \[ 8/8] dice=0.6179

&#x20; → Trial 013 DONE | dice=0.6179 | 75s



\[Trial 014/54] lr=0.00530  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3730

&#x20;   Epoch \[ 2/8] dice=0.3818

&#x20;   Epoch \[ 3/8] dice=0.4604

&#x20;   Epoch \[ 4/8] dice=0.4023

&#x20;   Epoch \[ 5/8] dice=0.4473

&#x20;   Epoch \[ 6/8] dice=0.5293

&#x20;   Epoch \[ 7/8] dice=0.5980

&#x20;   Epoch \[ 8/8] dice=0.6251

&#x20; → Trial 014 DONE | dice=0.6251 | 75s



\[Trial 015/54] lr=0.00148  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3524

&#x20;   Epoch \[ 2/8] dice=0.3873

&#x20;   Epoch \[ 3/8] dice=0.5745

&#x20;   Epoch \[ 4/8] dice=0.5741

&#x20;   Epoch \[ 5/8] dice=0.6203

&#x20;   Epoch \[ 6/8] dice=0.6348

&#x20;   Epoch \[ 7/8] dice=0.6670

&#x20;   Epoch \[ 8/8] dice=0.6877

&#x20; → Trial 015 DONE | dice=0.6877 | 71s



\[Trial 016/54] lr=0.00974  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.0120

&#x20;   Epoch \[ 2/8] dice=0.0987

&#x20;   Epoch \[ 3/8] dice=0.4125

&#x20;   Epoch \[ 4/8] dice=0.4892

&#x20;   Epoch \[ 5/8] dice=0.3882

&#x20;   Epoch \[ 6/8] dice=0.4968

&#x20;   Epoch \[ 7/8] dice=0.6074

&#x20;   Epoch \[ 8/8] dice=0.6479

&#x20; → Trial 016 DONE | dice=0.6479 | 75s



\[Trial 017/54] lr=0.00462  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1786

&#x20;   Epoch \[ 2/8] dice=0.4187

&#x20;   Epoch \[ 3/8] dice=0.3271

&#x20;   Epoch \[ 4/8] dice=0.4540

&#x20;   Epoch \[ 5/8] dice=0.5595

&#x20;   Epoch \[ 6/8] dice=0.6124

&#x20;   Epoch \[ 7/8] dice=0.6500

&#x20;   Epoch \[ 8/8] dice=0.6808

&#x20; → Trial 017 DONE | dice=0.6808 | 75s



\[Trial 018/54] lr=0.00622  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2187

&#x20;   Epoch \[ 2/8] dice=0.2946

&#x20;   Epoch \[ 3/8] dice=0.4167

&#x20;   Epoch \[ 4/8] dice=0.4019

&#x20;   Epoch \[ 5/8] dice=0.5227

&#x20;   Epoch \[ 6/8] dice=0.5974

&#x20;   Epoch \[ 7/8] dice=0.6316

&#x20;   Epoch \[ 8/8] dice=0.6715

&#x20; → Trial 018 DONE | dice=0.6715 | 81s



\[Trial 019/54] lr=0.00519  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2306

&#x20;   Epoch \[ 2/8] dice=0.3323

&#x20;   Epoch \[ 3/8] dice=0.4654

&#x20;   Epoch \[ 4/8] dice=0.2830

&#x20;   Epoch \[ 5/8] dice=0.3870

&#x20;   Epoch \[ 6/8] dice=0.4929

&#x20;   Epoch \[ 7/8] dice=0.6100

&#x20;   Epoch \[ 8/8] dice=0.6539

&#x20; → Trial 019 DONE | dice=0.6539 | 72s



\[Trial 020/54] lr=0.00472  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4076

&#x20;   Epoch \[ 2/8] dice=0.3657

&#x20;   Epoch \[ 3/8] dice=0.2897

&#x20;   Epoch \[ 4/8] dice=0.4969

&#x20;   Epoch \[ 5/8] dice=0.4261

&#x20;   Epoch \[ 6/8] dice=0.5796

&#x20;   Epoch \[ 7/8] dice=0.5919

&#x20;   Epoch \[ 8/8] dice=0.6727

&#x20; → Trial 020 DONE | dice=0.6727 | 75s



\[Trial 021/54] lr=0.00611  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3903

&#x20;   Epoch \[ 2/8] dice=0.1224

&#x20;   Epoch \[ 3/8] dice=0.3172

&#x20;   Epoch \[ 4/8] dice=0.5204

&#x20;   Epoch \[ 5/8] dice=0.5399

&#x20;   Epoch \[ 6/8] dice=0.5929

&#x20;   Epoch \[ 7/8] dice=0.6579

&#x20;   Epoch \[ 8/8] dice=0.6770

&#x20; → Trial 021 DONE | dice=0.6770 | 75s



\[Trial 022/54] lr=0.00456  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3561

&#x20;   Epoch \[ 2/8] dice=0.3460

&#x20;   Epoch \[ 3/8] dice=0.3190

&#x20;   Epoch \[ 4/8] dice=0.4865

&#x20;   Epoch \[ 5/8] dice=0.3579

&#x20;   Epoch \[ 6/8] dice=0.5682

&#x20;   Epoch \[ 7/8] dice=0.6076

&#x20;   Epoch \[ 8/8] dice=0.6312

&#x20; → Trial 022 DONE | dice=0.6312 | 71s



\[Trial 023/54] lr=0.00949  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3027

&#x20;   Epoch \[ 2/8] dice=0.4386

&#x20;   Epoch \[ 3/8] dice=0.4319

&#x20;   Epoch \[ 4/8] dice=0.4173

&#x20;   Epoch \[ 5/8] dice=0.4049

&#x20;   Epoch \[ 6/8] dice=0.2335

&#x20;   Epoch \[ 7/8] dice=0.4970

&#x20;   Epoch \[ 8/8] dice=0.5490

&#x20; → Trial 023 DONE | dice=0.5490 | 71s



\[Trial 024/54] lr=0.00568  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1980

&#x20;   Epoch \[ 2/8] dice=0.4384

&#x20;   Epoch \[ 3/8] dice=0.2934

&#x20;   Epoch \[ 4/8] dice=0.5514

&#x20;   Epoch \[ 5/8] dice=0.5081

&#x20;   Epoch \[ 6/8] dice=0.6138

&#x20;   Epoch \[ 7/8] dice=0.6228

&#x20;   Epoch \[ 8/8] dice=0.6593

&#x20; → Trial 024 DONE | dice=0.6593 | 71s



\[Trial 025/54] lr=0.00312  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2803

&#x20;   Epoch \[ 2/8] dice=0.1627

&#x20;   Epoch \[ 3/8] dice=0.5043

&#x20;   Epoch \[ 4/8] dice=0.5504

&#x20;   Epoch \[ 5/8] dice=0.4952

&#x20;   Epoch \[ 6/8] dice=0.6032

&#x20;   Epoch \[ 7/8] dice=0.6541

&#x20;   Epoch \[ 8/8] dice=0.6866

&#x20; → Trial 025 DONE | dice=0.6866 | 75s



\[Trial 026/54] lr=0.00239  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3330

&#x20;   Epoch \[ 2/8] dice=0.3143

&#x20;   Epoch \[ 3/8] dice=0.5115

&#x20;   Epoch \[ 4/8] dice=0.4788

&#x20;   Epoch \[ 5/8] dice=0.4690

&#x20;   Epoch \[ 6/8] dice=0.5137

&#x20;   Epoch \[ 7/8] dice=0.5927

&#x20;   Epoch \[ 8/8] dice=0.6371

&#x20; → Trial 026 DONE | dice=0.6371 | 71s



\[Trial 027/54] lr=0.00446  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1846

&#x20;   Epoch \[ 2/8] dice=0.2993

&#x20;   Epoch \[ 3/8] dice=0.4804

&#x20;   Epoch \[ 4/8] dice=0.4863

&#x20;   Epoch \[ 5/8] dice=0.4580

&#x20;   Epoch \[ 6/8] dice=0.5907

&#x20;   Epoch \[ 7/8] dice=0.6099

&#x20;   Epoch \[ 8/8] dice=0.6726

&#x20; → Trial 027 DONE | dice=0.6726 | 75s



\[Trial 028/54] lr=0.00614  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.0869

&#x20;   Epoch \[ 2/8] dice=0.1240

&#x20;   Epoch \[ 3/8] dice=0.4441

&#x20;   Epoch \[ 4/8] dice=0.2740

&#x20;   Epoch \[ 5/8] dice=0.4959

&#x20;   Epoch \[ 6/8] dice=0.2833

&#x20;   Epoch \[ 7/8] dice=0.5828

&#x20;   Epoch \[ 8/8] dice=0.5982

&#x20; → Trial 028 DONE | dice=0.5982 | 71s



\[Trial 029/54] lr=0.00044  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3483

&#x20;   Epoch \[ 2/8] dice=0.4958

&#x20;   Epoch \[ 3/8] dice=0.5130

&#x20;   Epoch \[ 4/8] dice=0.6303

&#x20;   Epoch \[ 5/8] dice=0.6170

&#x20;   Epoch \[ 6/8] dice=0.6556

&#x20;   Epoch \[ 7/8] dice=0.7070

&#x20;   Epoch \[ 8/8] dice=0.7169

&#x20; → Trial 029 DONE | dice=0.7169 | 71s



\[Trial 030/54] lr=0.00397  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2937

&#x20;   Epoch \[ 2/8] dice=0.4965

&#x20;   Epoch \[ 3/8] dice=0.4206

&#x20;   Epoch \[ 4/8] dice=0.4044

&#x20;   Epoch \[ 5/8] dice=0.5402

&#x20;   Epoch \[ 6/8] dice=0.5894

&#x20;   Epoch \[ 7/8] dice=0.6092

&#x20;   Epoch \[ 8/8] dice=0.6587

&#x20; → Trial 030 DONE | dice=0.6587 | 71s



\[Trial 031/54] lr=0.00666  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1993

&#x20;   Epoch \[ 2/8] dice=0.4373

&#x20;   Epoch \[ 3/8] dice=0.5045

&#x20;   Epoch \[ 4/8] dice=0.2589

&#x20;   Epoch \[ 5/8] dice=0.5888

&#x20;   Epoch \[ 6/8] dice=0.6283

&#x20;   Epoch \[ 7/8] dice=0.6026

&#x20;   Epoch \[ 8/8] dice=0.6724

&#x20; → Trial 031 DONE | dice=0.6724 | 72s



\[Trial 032/54] lr=0.00431  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1797

&#x20;   Epoch \[ 2/8] dice=0.3041

&#x20;   Epoch \[ 3/8] dice=0.4305

&#x20;   Epoch \[ 4/8] dice=0.4707

&#x20;   Epoch \[ 5/8] dice=0.5921

&#x20;   Epoch \[ 6/8] dice=0.3870

&#x20;   Epoch \[ 7/8] dice=0.6411

&#x20;   Epoch \[ 8/8] dice=0.6778

&#x20; → Trial 032 DONE | dice=0.6778 | 71s



\[Trial 033/54] lr=0.00551  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1913

&#x20;   Epoch \[ 2/8] dice=0.3047

&#x20;   Epoch \[ 3/8] dice=0.4172

&#x20;   Epoch \[ 4/8] dice=0.3660

&#x20;   Epoch \[ 5/8] dice=0.4292

&#x20;   Epoch \[ 6/8] dice=0.5105

&#x20;   Epoch \[ 7/8] dice=0.5893

&#x20;   Epoch \[ 8/8] dice=0.6630

&#x20; → Trial 033 DONE | dice=0.6630 | 71s



\[Trial 034/54] lr=0.00041  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3763

&#x20;   Epoch \[ 2/8] dice=0.4863

&#x20;   Epoch \[ 3/8] dice=0.5634

&#x20;   Epoch \[ 4/8] dice=0.6394

&#x20;   Epoch \[ 5/8] dice=0.6506

&#x20;   Epoch \[ 6/8] dice=0.6610

&#x20;   Epoch \[ 7/8] dice=0.7153

&#x20;   Epoch \[ 8/8] dice=0.7214

&#x20; → Trial 034 DONE | dice=0.7214 | 71s

&#x20; ★ New best! dice=0.7214



\[Trial 035/54] lr=0.00777  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2642

&#x20;   Epoch \[ 2/8] dice=0.2453

&#x20;   Epoch \[ 3/8] dice=0.0635

&#x20;   Epoch \[ 4/8] dice=0.4092

&#x20;   Epoch \[ 5/8] dice=0.4581

&#x20;   Epoch \[ 6/8] dice=0.4005

&#x20;   Epoch \[ 7/8] dice=0.6332

&#x20;   Epoch \[ 8/8] dice=0.6645

&#x20; → Trial 035 DONE | dice=0.6645 | 71s



\[Trial 036/54] lr=0.00401  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3741

&#x20;   Epoch \[ 2/8] dice=0.2330

&#x20;   Epoch \[ 3/8] dice=0.2027

&#x20;   Epoch \[ 4/8] dice=0.4845

&#x20;   Epoch \[ 5/8] dice=0.4692

&#x20;   Epoch \[ 6/8] dice=0.5974

&#x20;   Epoch \[ 7/8] dice=0.6564

&#x20;   Epoch \[ 8/8] dice=0.6692

&#x20; → Trial 036 DONE | dice=0.6692 | 71s



\[Trial 037/54] lr=0.00602  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1428

&#x20;   Epoch \[ 2/8] dice=0.0000

&#x20;   Epoch \[ 3/8] dice=0.4489

&#x20;   Epoch \[ 4/8] dice=0.5520

&#x20;   Epoch \[ 5/8] dice=0.5169

&#x20;   Epoch \[ 6/8] dice=0.5494

&#x20;   Epoch \[ 7/8] dice=0.6366

&#x20;   Epoch \[ 8/8] dice=0.6607

&#x20; → Trial 037 DONE | dice=0.6607 | 72s



\[Trial 038/54] lr=0.00333  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3164

&#x20;   Epoch \[ 2/8] dice=0.4743

&#x20;   Epoch \[ 3/8] dice=0.4765

&#x20;   Epoch \[ 4/8] dice=0.4974

&#x20;   Epoch \[ 5/8] dice=0.5421

&#x20;   Epoch \[ 6/8] dice=0.6217

&#x20;   Epoch \[ 7/8] dice=0.6478

&#x20;   Epoch \[ 8/8] dice=0.6627

&#x20; → Trial 038 DONE | dice=0.6627 | 72s



\[Trial 039/54] lr=0.00204  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3546

&#x20;   Epoch \[ 2/8] dice=0.2803

&#x20;   Epoch \[ 3/8] dice=0.3891

&#x20;   Epoch \[ 4/8] dice=0.5718

&#x20;   Epoch \[ 5/8] dice=0.5947

&#x20;   Epoch \[ 6/8] dice=0.6441

&#x20;   Epoch \[ 7/8] dice=0.6803

&#x20;   Epoch \[ 8/8] dice=0.6974

&#x20; → Trial 039 DONE | dice=0.6974 | 75s



\[Trial 040/54] lr=0.00962  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3980

&#x20;   Epoch \[ 2/8] dice=0.1211

&#x20;   Epoch \[ 3/8] dice=0.4595

&#x20;   Epoch \[ 4/8] dice=0.3887

&#x20;   Epoch \[ 5/8] dice=0.5012

&#x20;   Epoch \[ 6/8] dice=0.5234

&#x20;   Epoch \[ 7/8] dice=0.5015

&#x20;   Epoch \[ 8/8] dice=0.5978

&#x20; → Trial 040 DONE | dice=0.5978 | 75s



\[Trial 041/54] lr=0.00395  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3153

&#x20;   Epoch \[ 2/8] dice=0.1460

&#x20;   Epoch \[ 3/8] dice=0.4993

&#x20;   Epoch \[ 4/8] dice=0.4582

&#x20;   Epoch \[ 5/8] dice=0.5485

&#x20;   Epoch \[ 6/8] dice=0.6032

&#x20;   Epoch \[ 7/8] dice=0.6068

&#x20;   Epoch \[ 8/8] dice=0.6565

&#x20; → Trial 041 DONE | dice=0.6565 | 71s



\[Trial 042/54] lr=0.00544  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3272

&#x20;   Epoch \[ 2/8] dice=0.4039

&#x20;   Epoch \[ 3/8] dice=0.0470

&#x20;   Epoch \[ 4/8] dice=0.0851

&#x20;   Epoch \[ 5/8] dice=0.4957

&#x20;   Epoch \[ 6/8] dice=0.5486

&#x20;   Epoch \[ 7/8] dice=0.5989

&#x20;   Epoch \[ 8/8] dice=0.6576

&#x20; → Trial 042 DONE | dice=0.6576 | 71s



\[Trial 043/54] lr=0.00363  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3848

&#x20;   Epoch \[ 2/8] dice=0.3205

&#x20;   Epoch \[ 3/8] dice=0.3423

&#x20;   Epoch \[ 4/8] dice=0.5381

&#x20;   Epoch \[ 5/8] dice=0.5663

&#x20;   Epoch \[ 6/8] dice=0.6427

&#x20;   Epoch \[ 7/8] dice=0.6654

&#x20;   Epoch \[ 8/8] dice=0.6816

&#x20; → Trial 043 DONE | dice=0.6816 | 71s



\[Trial 044/54] lr=0.00611  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2376

&#x20;   Epoch \[ 2/8] dice=0.3656

&#x20;   Epoch \[ 3/8] dice=0.4388

&#x20;   Epoch \[ 4/8] dice=0.1838

&#x20;   Epoch \[ 5/8] dice=0.5174

&#x20;   Epoch \[ 6/8] dice=0.4713

&#x20;   Epoch \[ 7/8] dice=0.3823

&#x20;   Epoch \[ 8/8] dice=0.6040

&#x20; → Trial 044 DONE | dice=0.6040 | 75s



\[Trial 045/54] lr=0.00150  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3698

&#x20;   Epoch \[ 2/8] dice=0.5124

&#x20;   Epoch \[ 3/8] dice=0.5285

&#x20;   Epoch \[ 4/8] dice=0.5645

&#x20;   Epoch \[ 5/8] dice=0.5346

&#x20;   Epoch \[ 6/8] dice=0.6430

&#x20;   Epoch \[ 7/8] dice=0.6738

&#x20;   Epoch \[ 8/8] dice=0.6972

&#x20; → Trial 045 DONE | dice=0.6972 | 75s



\[Trial 046/54] lr=0.00174  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.3565

&#x20;   Epoch \[ 2/8] dice=0.4616

&#x20;   Epoch \[ 3/8] dice=0.5308

&#x20;   Epoch \[ 4/8] dice=0.4565

&#x20;   Epoch \[ 5/8] dice=0.6006

&#x20;   Epoch \[ 6/8] dice=0.6282

&#x20;   Epoch \[ 7/8] dice=0.6418

&#x20;   Epoch \[ 8/8] dice=0.6632

&#x20; → Trial 046 DONE | dice=0.6632 | 75s



\[Trial 047/54] lr=0.00987  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1923

&#x20;   Epoch \[ 2/8] dice=0.3926

&#x20;   Epoch \[ 3/8] dice=0.4915

&#x20;   Epoch \[ 4/8] dice=0.2992

&#x20;   Epoch \[ 5/8] dice=0.5127

&#x20;   Epoch \[ 6/8] dice=0.5098

&#x20;   Epoch \[ 7/8] dice=0.4988

&#x20;   Epoch \[ 8/8] dice=0.5868

&#x20; → Trial 047 DONE | dice=0.5868 | 71s



\[Trial 048/54] lr=0.00401  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2785

&#x20;   Epoch \[ 2/8] dice=0.3814

&#x20;   Epoch \[ 3/8] dice=0.4295

&#x20;   Epoch \[ 4/8] dice=0.5274

&#x20;   Epoch \[ 5/8] dice=0.4413

&#x20;   Epoch \[ 6/8] dice=0.5636

&#x20;   Epoch \[ 7/8] dice=0.5737

&#x20;   Epoch \[ 8/8] dice=0.6228

&#x20; → Trial 048 DONE | dice=0.6228 | 71s



\[Trial 049/54] lr=0.00015  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4520

&#x20;   Epoch \[ 2/8] dice=0.5824

&#x20;   Epoch \[ 3/8] dice=0.6074

&#x20;   Epoch \[ 4/8] dice=0.6391

&#x20;   Epoch \[ 5/8] dice=0.7032

&#x20;   Epoch \[ 6/8] dice=0.7284

&#x20;   Epoch \[ 7/8] dice=0.7358

&#x20;   Epoch \[ 8/8] dice=0.7483

&#x20; → Trial 049 DONE | dice=0.7483 | 75s

&#x20; ★ New best! dice=0.7483



\[Trial 050/54] lr=0.00207  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2061

&#x20;   Epoch \[ 2/8] dice=0.4289

&#x20;   Epoch \[ 3/8] dice=0.4265

&#x20;   Epoch \[ 4/8] dice=0.5477

&#x20;   Epoch \[ 5/8] dice=0.5075

&#x20;   Epoch \[ 6/8] dice=0.6078

&#x20;   Epoch \[ 7/8] dice=0.6015

&#x20;   Epoch \[ 8/8] dice=0.6626

&#x20; → Trial 050 DONE | dice=0.6626 | 72s



\[Trial 051/54] lr=0.00732  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.4017

&#x20;   Epoch \[ 2/8] dice=0.3096

&#x20;   Epoch \[ 3/8] dice=0.4793

&#x20;   Epoch \[ 4/8] dice=0.4999

&#x20;   Epoch \[ 5/8] dice=0.5131

&#x20;   Epoch \[ 6/8] dice=0.5429

&#x20;   Epoch \[ 7/8] dice=0.5664

&#x20;   Epoch \[ 8/8] dice=0.6292

&#x20; → Trial 051 DONE | dice=0.6292 | 76s



\[Trial 052/54] lr=0.00610  bs=16

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.1253

&#x20;   Epoch \[ 2/8] dice=0.3352

&#x20;   Epoch \[ 3/8] dice=0.4470

&#x20;   Epoch \[ 4/8] dice=0.5555

&#x20;   Epoch \[ 5/8] dice=0.5652

&#x20;   Epoch \[ 6/8] dice=0.5000

&#x20;   Epoch \[ 7/8] dice=0.6279

&#x20;   Epoch \[ 8/8] dice=0.6503

&#x20; → Trial 052 DONE | dice=0.6503 | 71s



\[Trial 053/54] lr=0.00365  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.0433

&#x20;   Epoch \[ 2/8] dice=0.2947

&#x20;   Epoch \[ 3/8] dice=0.4412

&#x20;   Epoch \[ 4/8] dice=0.5378

&#x20;   Epoch \[ 5/8] dice=0.5550

&#x20;   Epoch \[ 6/8] dice=0.6285

&#x20;   Epoch \[ 7/8] dice=0.6340

&#x20;   Epoch \[ 8/8] dice=0.6621

&#x20; → Trial 053 DONE | dice=0.6621 | 75s



\[Trial 054/54] lr=0.00916  bs=8

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=0

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   Epoch \[ 1/8] dice=0.2412

&#x20;   Epoch \[ 2/8] dice=0.1006

&#x20;   Epoch \[ 3/8] dice=0.2952

&#x20;   Epoch \[ 4/8] dice=0.4560

&#x20;   Epoch \[ 5/8] dice=0.4792

&#x20;   Epoch \[ 6/8] dice=0.4574

&#x20;   Epoch \[ 7/8] dice=0.5267

&#x20;   Epoch \[ 8/8] dice=0.5574

&#x20; → Trial 054 DONE | dice=0.5574 | 75s



==============================================================

\[RandomSearch] COMPLETE

&#x20; Best lr         : 0.00015

&#x20; Best batch\_size : 8

&#x20; Best Dice       : 0.7483

==============================================================



\[RandomSearch] Results saved → random\_search\_results.csv



\[RandomSearch] Top 5 trials:

&#x20;  Trial         LR   BS     Dice

&#x20; ----------------------------------

&#x20;     49    0.00015    8   0.7483

&#x20;     34    0.00041   16   0.7214

&#x20;     10    0.00011   16   0.7204

&#x20;     29    0.00044   16   0.7169

&#x20;      5    0.00068   16   0.7126













**Multi run evaluation**

(segmentation) D:\\Brain\_Segmentation>python multi\_run\_eval.py --data\_dir processed\_dataset --pso\_lr 0.00381 --pso\_bs 16 --epochs 30



\[Device] NVIDIA GeForce RTX 4060 Laptop GPU | VRAM: 8.0 GB

\[MultiRun] Seeds: \[42, 52] | Epochs per run: 30



==============================================================

\[MultiRun] Standard U-Net

&#x20; lr=0.00100 | batch\_size=16 | epochs=30

&#x20; Seeds: \[42, 52]

==============================================================



&#x20; ── Run 1/2 (seed=42) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[Standard U-Net | seed=42] Epoch \[  1/30] train=0.5376 | dice=0.4399 | best=0.4399 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  2/30] train=0.3915 | dice=0.4073 | best=0.4399 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  3/30] train=0.3140 | dice=0.4970 | best=0.4970 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  4/30] train=0.2762 | dice=0.4025 | best=0.4970 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  5/30] train=0.2389 | dice=0.6063 | best=0.6063 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  6/30] train=0.2299 | dice=0.5722 | best=0.6063 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  7/30] train=0.2269 | dice=0.5994 | best=0.6063 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  8/30] train=0.2064 | dice=0.6463 | best=0.6463 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  9/30] train=0.1913 | dice=0.6782 | best=0.6782 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 10/30] train=0.1894 | dice=0.6839 | best=0.6839 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 11/30] train=0.1799 | dice=0.5866 | best=0.6839 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 12/30] train=0.1704 | dice=0.6757 | best=0.6839 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 13/30] train=0.1672 | dice=0.7177 | best=0.7177 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 14/30] train=0.1627 | dice=0.7056 | best=0.7177 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 15/30] train=0.1606 | dice=0.7263 | best=0.7263 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 16/30] train=0.1516 | dice=0.7061 | best=0.7263 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 17/30] train=0.1447 | dice=0.7354 | best=0.7354 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 18/30] train=0.1436 | dice=0.7198 | best=0.7354 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 19/30] train=0.1385 | dice=0.7464 | best=0.7464 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 20/30] train=0.1336 | dice=0.7462 | best=0.7464 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 21/30] train=0.1313 | dice=0.7527 | best=0.7527 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 22/30] train=0.1241 | dice=0.7689 | best=0.7689 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 23/30] train=0.1212 | dice=0.7757 | best=0.7757 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 24/30] train=0.1159 | dice=0.7826 | best=0.7826 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 25/30] train=0.1141 | dice=0.7805 | best=0.7826 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 26/30] train=0.1102 | dice=0.7826 | best=0.7826 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 27/30] train=0.1094 | dice=0.7868 | best=0.7868 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 28/30] train=0.1070 | dice=0.7836 | best=0.7868 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 29/30] train=0.1057 | dice=0.7894 | best=0.7894 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 30/30] train=0.1035 | dice=0.7899 | best=0.7899 | VRAM=486MB

&#x20; Run 1 final Dice = 0.7899



&#x20; ── Run 2/2 (seed=52) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  1/30] train=0.4719 | dice=0.0319 | best=0.0319 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  2/30] train=0.3477 | dice=0.2950 | best=0.2950 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  3/30] train=0.3128 | dice=0.2982 | best=0.2982 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  4/30] train=0.2856 | dice=0.5156 | best=0.5156 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  5/30] train=0.2677 | dice=0.4048 | best=0.5156 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  6/30] train=0.2479 | dice=0.5315 | best=0.5315 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  7/30] train=0.2484 | dice=0.5953 | best=0.5953 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  8/30] train=0.2407 | dice=0.6641 | best=0.6641 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  9/30] train=0.2232 | dice=0.5969 | best=0.6641 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 10/30] train=0.2193 | dice=0.5028 | best=0.6641 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 11/30] train=0.2062 | dice=0.5535 | best=0.6641 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 12/30] train=0.2002 | dice=0.6205 | best=0.6641 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 13/30] train=0.1927 | dice=0.7147 | best=0.7147 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 14/30] train=0.1802 | dice=0.6181 | best=0.7147 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 15/30] train=0.1768 | dice=0.6812 | best=0.7147 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 16/30] train=0.1691 | dice=0.6950 | best=0.7147 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 17/30] train=0.1618 | dice=0.7219 | best=0.7219 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 18/30] train=0.1619 | dice=0.7182 | best=0.7219 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 19/30] train=0.1499 | dice=0.7382 | best=0.7382 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 20/30] train=0.1462 | dice=0.7478 | best=0.7478 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 21/30] train=0.1403 | dice=0.7197 | best=0.7478 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 22/30] train=0.1306 | dice=0.7833 | best=0.7833 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 23/30] train=0.1302 | dice=0.7575 | best=0.7833 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 24/30] train=0.1242 | dice=0.7698 | best=0.7833 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 25/30] train=0.1164 | dice=0.7913 | best=0.7913 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 26/30] train=0.1127 | dice=0.7891 | best=0.7913 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 27/30] train=0.1102 | dice=0.7937 | best=0.7937 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 28/30] train=0.1068 | dice=0.7912 | best=0.7937 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 29/30] train=0.1047 | dice=0.7928 | best=0.7937 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 30/30] train=0.1037 | dice=0.7919 | best=0.7937 | VRAM=485MB

&#x20; Run 2 final Dice = 0.7937



\[MultiRun] U-Net + PSO COMPLETE

&#x20; Per-seed scores : \['0.7790', '0.7937']

&#x20; Dice = 0.7863 ± 0.0074



==============================================================

&#x20; Method                     Dice Mean   Dice Std  Scores

==============================================================

&#x20; Standard U-Net                0.7906     0.0007  \[0.7899  0.7913]

&#x20; U-Net + PSO                   0.7863     0.0074  \[0.7790  0.7937]

==============================================================



\[MultiRun] Publication-ready results:

&#x20; Standard U-Net: Dice = 0.7906 ± 0.0007

&#x20; U-Net + PSO: Dice = 0.7863 ± 0.0074



\[MultiRun] Results saved → multi\_run\_results.txt













**Multi-run Unet+PSO**

(segmentation) D:\\Brain\_Segmentation>python multi\_run\_eval.py --data\_dir processed\_dataset --pso\_lr 0.00015 --pso\_bs 8 --epochs 30



\[Device] NVIDIA GeForce RTX 4060 Laptop GPU | VRAM: 8.0 GB

\[MultiRun] Seeds: \[42, 52] | Epochs per run: 30



==============================================================

\[MultiRun] Standard U-Net

&#x20; lr=0.00100 | batch\_size=16 | epochs=30

&#x20; Seeds: \[42, 52]

==============================================================



&#x20; ── Run 1/2 (seed=42) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[Standard U-Net | seed=42] Epoch \[  1/30] train=0.5387 | dice=0.3609 | best=0.3609 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  2/30] train=0.3972 | dice=0.4322 | best=0.4322 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  3/30] train=0.3132 | dice=0.5336 | best=0.5336 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  4/30] train=0.2674 | dice=0.3734 | best=0.5336 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  5/30] train=0.2450 | dice=0.5983 | best=0.5983 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  6/30] train=0.2394 | dice=0.6001 | best=0.6001 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  7/30] train=0.2261 | dice=0.5878 | best=0.6001 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  8/30] train=0.2055 | dice=0.6247 | best=0.6247 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[  9/30] train=0.2023 | dice=0.6659 | best=0.6659 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 10/30] train=0.1956 | dice=0.6766 | best=0.6766 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 11/30] train=0.1820 | dice=0.6925 | best=0.6925 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 12/30] train=0.1769 | dice=0.6639 | best=0.6925 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 13/30] train=0.1636 | dice=0.6955 | best=0.6955 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 14/30] train=0.1675 | dice=0.6988 | best=0.6988 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 15/30] train=0.1607 | dice=0.7332 | best=0.7332 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 16/30] train=0.1512 | dice=0.7310 | best=0.7332 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 17/30] train=0.1503 | dice=0.7420 | best=0.7420 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 18/30] train=0.1407 | dice=0.7667 | best=0.7667 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 19/30] train=0.1390 | dice=0.7710 | best=0.7710 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 20/30] train=0.1297 | dice=0.7518 | best=0.7710 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 21/30] train=0.1317 | dice=0.7753 | best=0.7753 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 22/30] train=0.1217 | dice=0.7725 | best=0.7753 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 23/30] train=0.1198 | dice=0.7773 | best=0.7773 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 24/30] train=0.1167 | dice=0.7945 | best=0.7945 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 25/30] train=0.1134 | dice=0.7958 | best=0.7958 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 26/30] train=0.1098 | dice=0.7931 | best=0.7958 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 27/30] train=0.1063 | dice=0.7930 | best=0.7958 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 28/30] train=0.1045 | dice=0.7981 | best=0.7981 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 29/30] train=0.1027 | dice=0.7994 | best=0.7994 | VRAM=486MB

&#x20;   \[Standard U-Net | seed=42] Epoch \[ 30/30] train=0.1014 | dice=0.8000 | best=0.8000 | VRAM=486MB

&#x20; Run 1 final Dice = 0.8000



&#x20; ── Run 2/2 (seed=52) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=16 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[Standard U-Net | seed=52] Epoch \[  1/30] train=0.5676 | dice=0.2881 | best=0.2881 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  2/30] train=0.4154 | dice=0.2894 | best=0.2894 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  3/30] train=0.3317 | dice=0.3275 | best=0.3275 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  4/30] train=0.2843 | dice=0.5809 | best=0.5809 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  5/30] train=0.2642 | dice=0.5174 | best=0.5809 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  6/30] train=0.2551 | dice=0.6152 | best=0.6152 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  7/30] train=0.2392 | dice=0.6167 | best=0.6167 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  8/30] train=0.2327 | dice=0.5734 | best=0.6167 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[  9/30] train=0.2335 | dice=0.6309 | best=0.6309 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 10/30] train=0.2194 | dice=0.6303 | best=0.6309 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 11/30] train=0.2057 | dice=0.6769 | best=0.6769 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 12/30] train=0.1947 | dice=0.6121 | best=0.6769 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 13/30] train=0.1886 | dice=0.6439 | best=0.6769 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 14/30] train=0.1847 | dice=0.6641 | best=0.6769 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 15/30] train=0.1758 | dice=0.6876 | best=0.6876 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 16/30] train=0.1772 | dice=0.7317 | best=0.7317 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 17/30] train=0.1623 | dice=0.7055 | best=0.7317 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 18/30] train=0.1550 | dice=0.7280 | best=0.7317 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 19/30] train=0.1520 | dice=0.7413 | best=0.7413 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 20/30] train=0.1422 | dice=0.7538 | best=0.7538 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 21/30] train=0.1362 | dice=0.7556 | best=0.7556 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 22/30] train=0.1316 | dice=0.7605 | best=0.7605 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 23/30] train=0.1283 | dice=0.7726 | best=0.7726 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 24/30] train=0.1233 | dice=0.7653 | best=0.7726 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 25/30] train=0.1202 | dice=0.7824 | best=0.7824 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 26/30] train=0.1154 | dice=0.7820 | best=0.7824 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 27/30] train=0.1153 | dice=0.7874 | best=0.7874 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 28/30] train=0.1118 | dice=0.7874 | best=0.7874 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 29/30] train=0.1125 | dice=0.7935 | best=0.7935 | VRAM=485MB

&#x20;   \[Standard U-Net | seed=52] Epoch \[ 30/30] train=0.1088 | dice=0.7932 | best=0.7935 | VRAM=485MB

&#x20; Run 2 final Dice = 0.7935



\[MultiRun] Standard U-Net COMPLETE

&#x20; Per-seed scores : \['0.8000', '0.7935']

&#x20; Dice = 0.7967 ± 0.0032



==============================================================

\[MultiRun] U-Net + PSO

&#x20; lr=0.00015 | batch\_size=8 | epochs=30

&#x20; Seeds: \[42, 52]

==============================================================



&#x20; ── Run 1/2 (seed=42) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  1/30] train=0.5765 | dice=0.5018 | best=0.5018 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  2/30] train=0.4795 | dice=0.4586 | best=0.5018 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  3/30] train=0.4128 | dice=0.6566 | best=0.6566 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  4/30] train=0.3550 | dice=0.6620 | best=0.6620 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  5/30] train=0.3036 | dice=0.6528 | best=0.6620 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  6/30] train=0.2678 | dice=0.6871 | best=0.6871 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  7/30] train=0.2321 | dice=0.6983 | best=0.6983 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  8/30] train=0.2077 | dice=0.7043 | best=0.7043 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[  9/30] train=0.1872 | dice=0.7404 | best=0.7404 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 10/30] train=0.1710 | dice=0.7567 | best=0.7567 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 11/30] train=0.1570 | dice=0.7374 | best=0.7567 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 12/30] train=0.1462 | dice=0.7665 | best=0.7665 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 13/30] train=0.1422 | dice=0.7792 | best=0.7792 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 14/30] train=0.1385 | dice=0.7741 | best=0.7792 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 15/30] train=0.1289 | dice=0.7873 | best=0.7873 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 16/30] train=0.1209 | dice=0.7831 | best=0.7873 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 17/30] train=0.1139 | dice=0.7937 | best=0.7937 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 18/30] train=0.1081 | dice=0.7919 | best=0.7937 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 19/30] train=0.1069 | dice=0.7962 | best=0.7962 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 20/30] train=0.1004 | dice=0.8008 | best=0.8008 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 21/30] train=0.0966 | dice=0.8102 | best=0.8102 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 22/30] train=0.0969 | dice=0.8085 | best=0.8102 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 23/30] train=0.0928 | dice=0.8117 | best=0.8117 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 24/30] train=0.0886 | dice=0.8154 | best=0.8154 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 25/30] train=0.0859 | dice=0.8193 | best=0.8193 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 26/30] train=0.0835 | dice=0.8197 | best=0.8197 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 27/30] train=0.0822 | dice=0.8210 | best=0.8210 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 28/30] train=0.0813 | dice=0.8186 | best=0.8210 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 29/30] train=0.0800 | dice=0.8195 | best=0.8210 | VRAM=488MB

&#x20;   \[U-Net + PSO | seed=42] Epoch \[ 30/30] train=0.0801 | dice=0.8211 | best=0.8211 | VRAM=488MB

&#x20; Run 1 final Dice = 0.8211



&#x20; ── Run 2/2 (seed=52) ──────────────────────

\[Dataset] Total: 1373 | Train: 1099 | Val: 274 | batch=8 | workers=4

\[Model] U-Net | Parameters: 31,383,681 | dropout=0.000

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  1/30] train=0.6107 | dice=0.4124 | best=0.4124 | VRAM=485MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  2/30] train=0.5096 | dice=0.6249 | best=0.6249 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  3/30] train=0.4410 | dice=0.6202 | best=0.6249 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  4/30] train=0.3835 | dice=0.6198 | best=0.6249 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  5/30] train=0.3328 | dice=0.7034 | best=0.7034 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  6/30] train=0.2856 | dice=0.6706 | best=0.7034 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  7/30] train=0.2464 | dice=0.7127 | best=0.7127 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  8/30] train=0.2247 | dice=0.7199 | best=0.7199 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[  9/30] train=0.1996 | dice=0.7410 | best=0.7410 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 10/30] train=0.1784 | dice=0.7644 | best=0.7644 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 11/30] train=0.1672 | dice=0.7495 | best=0.7644 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 12/30] train=0.1547 | dice=0.7540 | best=0.7644 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 13/30] train=0.1485 | dice=0.7519 | best=0.7644 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 14/30] train=0.1438 | dice=0.7546 | best=0.7644 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 15/30] train=0.1312 | dice=0.7746 | best=0.7746 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 16/30] train=0.1290 | dice=0.7823 | best=0.7823 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 17/30] train=0.1197 | dice=0.8116 | best=0.8116 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 18/30] train=0.1148 | dice=0.8032 | best=0.8116 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 19/30] train=0.1133 | dice=0.7917 | best=0.8116 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 20/30] train=0.1069 | dice=0.8097 | best=0.8116 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 21/30] train=0.1044 | dice=0.7940 | best=0.8116 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 22/30] train=0.1006 | dice=0.8159 | best=0.8159 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 23/30] train=0.0970 | dice=0.8248 | best=0.8248 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 24/30] train=0.0942 | dice=0.8216 | best=0.8248 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 25/30] train=0.0916 | dice=0.8216 | best=0.8248 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 26/30] train=0.0902 | dice=0.8294 | best=0.8294 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 27/30] train=0.0895 | dice=0.8306 | best=0.8306 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 28/30] train=0.0880 | dice=0.8289 | best=0.8306 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 29/30] train=0.0868 | dice=0.8287 | best=0.8306 | VRAM=487MB

&#x20;   \[U-Net + PSO | seed=52] Epoch \[ 30/30] train=0.0863 | dice=0.8251 | best=0.8306 | VRAM=487MB

&#x20; Run 2 final Dice = 0.8306



\[MultiRun] U-Net + PSO COMPLETE

&#x20; Per-seed scores : \['0.8211', '0.8306']

&#x20; Dice = 0.8258 ± 0.0047



==============================================================

&#x20; Method                     Dice Mean   Dice Std  Scores

==============================================================

&#x20; Standard U-Net                0.7967     0.0032  \[0.8000  0.7935]

&#x20; U-Net + PSO                   0.8258     0.0047  \[0.8211  0.8306]

==============================================================



\[MultiRun] Publication-ready results:

&#x20; Standard U-Net: Dice = 0.7967 ± 0.0032

&#x20; U-Net + PSO: Dice = 0.8258 ± 0.0047



\[MultiRun] Results saved → multi\_run\_results.txt







**Threshold\_vs\_Dice:**

threshold,dice

0.1,0.836603

0.15,0.839073

0.2,0.839599

0.25,0.840334

0.3,0.840581

0.35,0.840789

0.4,0.841342

0.45,0.841503

0.5,0.841457

0.55,0.841045

0.6,0.840911

0.65,0.84022

0.7,0.839433

0.75,0.838966

0.8,0.837956

0.85,0.836532

0.9,0.834156



