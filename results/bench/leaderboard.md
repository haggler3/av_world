## Absolute tier: fixed bars, 18 scenarios (11 core + 7 long-tail) x 8 checks

| policy | checks passed (of 144) | core | long-tail | scenarios cleared | most failed |
|---|---|---|---|---|---|
| PPO + barrier teacher, then no filter (phase 2) | 123 | 77 | 46 | 4 | safer unaided (12), safer than people (5), right of way (2), inside the barrier (2) |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 120 | 75 | 45 | 5 | safer unaided (12), safer than people (6), right of way (5), inside the barrier (1) |
| PPO + barrier teacher (phase 1) | 115 | 73 | 42 | 3 | safer unaided (13), safer than people (7), right of way (6), inside the barrier (2), lawfu |
| PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 112 | 70 | 42 | 2 | safer unaided (13), safer than people (8), right of way (6), lawful (3), keeps up (1), ins |
| PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 108 | 66 | 42 | 3 | safer unaided (12), safer than people (10), right of way (8), inside the barrier (3), lawf |
| PPO + barrier teacher (phase 1, GPU retrain) | 107 | 66 | 41 | 1 | safer unaided (16), safer than people (10), right of way (7), inside the barrier (3), lawf |
| rule-based AV | 104 | 69 | 35 | 0 | comfortable (16), right of way (8), safer than people (6), safer unaided (6), lawful (2),  |
| PPO, full world, 10x envelope penalty | 104 | 66 | 38 | 0 | keeps up (18), safer unaided (10), right of way (8), safer than people (4) |
| PPO, 234k steps (xnet3) | 76 | 48 | 28 | 0 | safer unaided (18), inside the barrier (18), light touch (15), safer than people (11), rig |
| PPO, 130k steps (xnet) | 54 | 37 | 17 | 0 | safer unaided (18), inside the barrier (18), light touch (17), right of way (15), lawful ( |
| imitation (BC) | 48 | 32 | 16 | 0 | light touch (18), safer unaided (18), right of way (18), inside the barrier (18), lawful ( |
| PPO, AVs-only world | 48 | 31 | 17 | 0 | light touch (18), safer unaided (18), lawful (18), right of way (18), inside the barrier ( |
| PPO, full world | 47 | 31 | 16 | 0 | light touch (18), safer unaided (18), lawful (18), right of way (18), inside the barrier ( |
| PPO, full world, no imitation | 46 | 29 | 17 | 0 | light touch (18), safer unaided (18), lawful (18), right of way (18), inside the barrier ( |
| PPO, failure-focused replay | 45 | 29 | 16 | 0 | light touch (18), safer unaided (18), lawful (18), right of way (18), inside the barrier ( |

Latent world-model planner, on the scenarios it ran (nominal, red_runners, inattentive), against every policy on the same ones:

| policy | absolute checks (of 24) | av_world score | crashes /AV-h unaided | harm /AV-h unaided | comfort pass | progress |
|---|---|---|---|---|---|---|
| rule-based AV | 18 | 78.2 | 0.008 | 0.008 | 0.903 | 1.059 |
| PPO + barrier teacher, then no filter (phase 2) | 23 | 77.9 | 0.033 | 0.050 | 0.986 | 1.024 |
| PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 21 | 72.0 | 0.008 | 0.080 | 0.983 | 1.049 |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 22 | 63.8 | 0.025 | 0.099 | 0.988 | 1.005 |
| PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 21 | 63.6 | 0.024 | 0.096 | 0.983 | 1.035 |
| PPO + barrier teacher (phase 1, GPU retrain) | 21 | 61.4 | 0.016 | 0.088 | 0.982 | 1.068 |
| PPO, full world, 10x envelope penalty | 19 | 56.9 | 0.000 | 0.000 | 0.985 | 0.665 |
| PPO + barrier teacher (phase 1) | 21 | 48.4 | 0.025 | 0.116 | 0.986 | 1.023 |
| latent world-model planner v2 (wm-mpc) | 17 | 46.0 | 0.039 | 0.211 | 0.980 | 0.949 |
| PPO, 234k steps (xnet3) | 13 | 7.9 | 0.640 | 1.092 | 0.981 | 1.065 |
| latent world-model planner (wm-mpc) | 8 | 0.8 | 0.471 | 0.829 | 0.958 | 1.052 |
| PPO, 130k steps (xnet) | 12 | 0.8 | 2.461 | 10.807 | 0.988 | 1.031 |
| imitation (BC) | 8 | 0.5 | 1.832 | 3.848 | 0.964 | 1.066 |
| PPO, full world, no imitation | 7 | 0.0 | 61.742 | 151.678 | 0.971 | 1.070 |
| PPO, AVs-only world | 9 | 0.0 | 72.151 | 175.980 | 0.977 | 1.086 |
| PPO, failure-focused replay | 8 | 0.0 | 71.186 | 172.749 | 0.974 | 1.072 |
| PPO, full world | 8 | 0.0 | 80.175 | 192.377 | 0.973 | 1.069 |

## Scorecard: the numbers to optimise (means over scenarios; worst = the lowest long-tail scenario)

| policy | av_world score: core | long-tail | worst | driving score | crashes /AV-h | unaided | crash UCB95 | harm unaided | comfort pass | jerk rms | progress | filter share | barrier exits | veto s /AV-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule-based AV | 77.2 | 39.0 | 2.8 | 96.8 | 0.079 | 0.079 | 0.187 | 0.156 | 0.914 | 2.70 | 1.115 | 0.0% | 0.12% | 13.2 |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 56.8 | 40.9 | 4.5 | 97.3 | 0.072 | 0.102 | 0.177 | 0.326 | 0.986 | 1.87 | 1.059 | 0.9% | 0.12% | 14.1 |
| PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 48.7 | 37.3 | 4.0 | 96.8 | 0.079 | 0.102 | 0.189 | 0.404 | 0.982 | 2.00 | 1.083 | 1.7% | 0.14% | 13.4 |
| PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 46.6 | 36.7 | 4.8 | 96.6 | 0.081 | 0.100 | 0.192 | 0.348 | 0.982 | 1.98 | 1.071 | 1.5% | 0.12% | 14.7 |
| PPO + barrier teacher, then no filter (phase 2) | 50.3 | 30.5 | 2.6 | 97.6 | 0.069 | 0.122 | 0.173 | 0.602 | 0.984 | 1.87 | 1.081 | 1.3% | 0.13% | 14.3 |
| PPO, full world, 10x envelope penalty | 49.2 | 25.9 | 5.5 | 68.3 | 0.036 | 0.061 | 0.126 | 0.178 | 0.983 | 1.12 | 0.699 | 0.3% | 0.03% | 8.4 |
| PPO + barrier teacher (phase 1) | 42.9 | 22.1 | 2.7 | 97.6 | 0.063 | 0.124 | 0.165 | 0.665 | 0.984 | 1.94 | 1.075 | 1.1% | 0.15% | 13.4 |
| PPO + barrier teacher (phase 1, GPU retrain) | 35.7 | 22.6 | 1.8 | 97.1 | 0.080 | 0.151 | 0.191 | 0.793 | 0.982 | 1.98 | 1.106 | 1.6% | 0.15% | 13.0 |
| PPO, 234k steps (xnet3) | 6.1 | 5.4 | 0.8 | 97.3 | 0.076 | 0.707 | 0.187 | 2.494 | 0.980 | 1.88 | 1.124 | 2.3% | 1.17% | 13.5 |
| imitation (BC) | 0.4 | 0.4 | 0.1 | 98.0 | 0.048 | 2.065 | 0.150 | 6.895 | 0.964 | 1.73 | 1.118 | 19.0% | 3.75% | 12.6 |
| PPO, 130k steps (xnet) | 0.5 | 0.2 | 0.0 | 98.3 | 0.043 | 2.949 | 0.144 | 16.355 | 0.987 | 1.49 | 1.084 | 2.6% | 1.50% | 12.7 |
| PPO, full world, no imitation | 0.0 | 0.0 | 0.0 | 97.9 | 0.056 | 63.066 | 0.161 | 169.856 | 0.973 | 1.66 | 1.122 | 15.8% | 5.66% | 13.8 |
| PPO, failure-focused replay | 0.0 | 0.0 | 0.0 | 97.7 | 0.063 | 70.522 | 0.170 | 185.648 | 0.974 | 1.62 | 1.124 | 20.6% | 6.55% | 11.9 |
| PPO, AVs-only world | 0.0 | 0.0 | 0.0 | 97.5 | 0.068 | 73.362 | 0.173 | 197.957 | 0.977 | 1.63 | 1.136 | 19.3% | 6.34% | 12.6 |
| PPO, full world | 0.0 | 0.0 | 0.0 | 97.9 | 0.057 | 80.336 | 0.161 | 213.051 | 0.974 | 1.64 | 1.123 | 19.4% | 6.92% | 12.0 |

## Relative tier: against the rule-based AV on the same traffic, 11 core scenarios x 8 checks

| policy | checks passed (of 88) | scenarios overcome (of 11) | nominal | rush_hour | red_runners | inattentive | jaywalkers | blockages | av_majority | cyclists | wet_road | squirrel_hill | downtown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rule-based AV | 88 | 11 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| PPO, 234k steps (xnet3) | 67 | 0 | 7/8 | 5/8 | 5/8 | 6/8 | 7/8 | 7/8 | 5/8 | 6/8 | 7/8 | 6/8 | 6/8 |
| PPO + barrier teacher, then no filter (phase 2) | 67 | 2 | 7/8 | 6/8 | 8/8 | 5/8 | 5/8 | 8/8 | 5/8 | 6/8 | 6/8 | 6/8 | 5/8 |
| PPO + barrier teacher (phase 1) | 66 | 1 | 6/8 | 6/8 | 7/8 | 6/8 | 7/8 | 8/8 | 3/8 | 7/8 | 6/8 | 5/8 | 5/8 |
| PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 66 | 1 | 8/8 | 6/8 | 5/8 | 7/8 | 6/8 | 6/8 | 7/8 | 6/8 | 5/8 | 5/8 | 5/8 |
| PPO + barrier teacher (phase 1, GPU retrain) | 63 | 0 | 7/8 | 6/8 | 6/8 | 5/8 | 6/8 | 5/8 | 5/8 | 7/8 | 6/8 | 6/8 | 4/8 |
| PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 63 | 1 | 8/8 | 7/8 | 7/8 | 4/8 | 7/8 | 4/8 | 5/8 | 4/8 | 7/8 | 6/8 | 4/8 |
| PPO + barrier teacher, speed + jaywalkers (phase 3) | 61 | 0 | 7/8 | 6/8 | 6/8 | 5/8 | 4/8 | 6/8 | 4/8 | 7/8 | 6/8 | 4/8 | 6/8 |
| PPO, full world, 10x envelope penalty | 59 | 0 | 5/8 | 5/8 | 6/8 | 6/8 | 6/8 | 6/8 | 5/8 | 5/8 | 5/8 | 4/8 | 6/8 |
| imitation (BC) | 45 | 0 | 4/8 | 4/8 | 3/8 | 4/8 | 5/8 | 4/8 | 4/8 | 4/8 | 4/8 | 5/8 | 4/8 |
| PPO, 130k steps (xnet) | 44 | 0 | 3/8 | 3/8 | 6/8 | 5/8 | 4/8 | 5/8 | 3/8 | 3/8 | 4/8 | 5/8 | 3/8 |
| PPO, full world | 40 | 0 | 4/8 | 4/8 | 3/8 | 4/8 | 2/8 | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 | 3/8 |
| PPO, AVs-only world | 38 | 0 | 4/8 | 4/8 | 4/8 | 2/8 | 2/8 | 4/8 | 3/8 | 4/8 | 4/8 | 4/8 | 3/8 |
| PPO, full world, no imitation | 38 | 0 | 4/8 | 3/8 | 3/8 | 3/8 | 2/8 | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 | 3/8 |
| PPO, failure-focused replay | 38 | 0 | 4/8 | 2/8 | 4/8 | 3/8 | 2/8 | 4/8 | 4/8 | 4/8 | 4/8 | 4/8 | 3/8 |


| scenario | policy | crashes /AV-h | raw crashes /AV-h | raw near misses /AV-h | hard brakes /AV-h | AV km/h | human km/h | envelope cuts |
|---|---|---|---|---|---|---|---|---|
| nominal | rule-based AV | 0.025 | 0.025 | 0.00 | 18.2 | 17.3 | 16.3 | 0.0% |
| nominal | imitation (BC) | 0.000 | 1.904 | 0.19 | 6.7 | 17.5 | 16.4 | 17.7% |
| nominal | PPO, AVs-only world | 0.000 | 73.636 | 3.69 | 4.4 | 17.7 | 16.3 | 18.5% |
| nominal | PPO, full world | 0.000 | 81.109 | 2.95 | 5.0 | 17.5 | 16.2 | 19.9% |
| nominal | PPO, full world, no imitation | 0.000 | 63.241 | 2.87 | 5.5 | 17.5 | 16.3 | 16.1% |
| nominal | PPO, failure-focused replay | 0.000 | 71.795 | 3.57 | 4.6 | 17.4 | 16.2 | 21.7% |
| nominal | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.02 | 2.7 | 9.6 | 14.5 | 0.2% |
| nominal | PPO, 130k steps (xnet) | 0.000 | 2.385 | 0.03 | 2.1 | 16.6 | 16.2 | 2.6% |
| nominal | PPO, 234k steps (xnet3) | 0.000 | 0.537 | 0.00 | 3.5 | 17.3 | 16.3 | 2.2% |
| nominal | PPO + barrier teacher (phase 1) | 0.000 | 0.000 | 0.02 | 2.5 | 16.7 | 16.3 | 1.1% |
| nominal | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.000 | 0.00 | 2.5 | 16.7 | 16.3 | 1.2% |
| nominal | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.000 | 0.00 | 2.0 | 16.4 | 16.2 | 0.9% |
| nominal | PPO + barrier teacher (phase 1, GPU retrain) | 0.000 | 0.000 | 0.02 | 3.4 | 17.7 | 16.4 | 1.6% |
| nominal | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.000 | 0.000 | 0.00 | 3.3 | 17.3 | 16.4 | 1.7% |
| nominal | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.000 | 0.00 | 3.1 | 17.1 | 16.4 | 1.6% |
| rush_hour | rule-based AV | 0.000 | 0.000 | 0.03 | 15.9 | 14.3 | 13.6 | 0.0% |
| rush_hour | imitation (BC) | 0.000 | 2.148 | 0.17 | 5.8 | 14.3 | 13.6 | 18.1% |
| rush_hour | PPO, AVs-only world | 0.000 | 70.655 | 3.04 | 3.8 | 14.5 | 13.6 | 16.9% |
| rush_hour | PPO, full world | 0.000 | 74.105 | 2.60 | 4.5 | 14.3 | 13.6 | 15.8% |
| rush_hour | PPO, full world, no imitation | 0.017 | 55.815 | 2.50 | 4.3 | 14.1 | 13.5 | 12.6% |
| rush_hour | PPO, failure-focused replay | 0.035 | 62.065 | 2.19 | 4.4 | 14.2 | 13.5 | 16.5% |
| rush_hour | PPO, full world, 10x envelope penalty | 0.000 | 0.017 | 0.02 | 2.7 | 8.4 | 11.8 | 0.2% |
| rush_hour | PPO, 130k steps (xnet) | 0.000 | 2.317 | 0.07 | 2.1 | 13.7 | 13.4 | 2.1% |
| rush_hour | PPO, 234k steps (xnet3) | 0.017 | 0.419 | 0.03 | 3.1 | 14.3 | 13.6 | 1.9% |
| rush_hour | PPO + barrier teacher (phase 1) | 0.000 | 0.017 | 0.02 | 2.3 | 13.8 | 13.6 | 0.9% |
| rush_hour | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.017 | 0.00 | 2.4 | 13.9 | 13.6 | 1.1% |
| rush_hour | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.017 | 0.02 | 2.6 | 13.7 | 13.5 | 0.8% |
| rush_hour | PPO + barrier teacher (phase 1, GPU retrain) | 0.017 | 0.017 | 0.02 | 2.6 | 14.4 | 13.6 | 1.5% |
| rush_hour | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.017 | 0.017 | 0.03 | 2.5 | 14.1 | 13.6 | 1.6% |
| rush_hour | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.000 | 0.02 | 2.5 | 13.9 | 13.5 | 1.4% |
| red_runners | rule-based AV | 0.000 | 0.000 | 0.00 | 18.4 | 17.2 | 16.2 | 0.0% |
| red_runners | imitation (BC) | 0.050 | 1.744 | 0.18 | 6.2 | 17.2 | 16.2 | 18.0% |
| red_runners | PPO, AVs-only world | 0.000 | 71.275 | 3.63 | 4.1 | 17.5 | 16.2 | 19.0% |
| red_runners | PPO, full world | 0.050 | 80.821 | 3.60 | 4.9 | 17.1 | 16.2 | 19.6% |
| red_runners | PPO, full world, no imitation | 0.050 | 61.342 | 3.71 | 5.5 | 17.2 | 16.1 | 15.8% |
| red_runners | PPO, failure-focused replay | 0.000 | 70.634 | 3.73 | 5.1 | 17.2 | 16.2 | 21.7% |
| red_runners | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 2.9 | 9.5 | 14.3 | 0.2% |
| red_runners | PPO, 130k steps (xnet) | 0.025 | 2.613 | 0.00 | 2.3 | 16.8 | 16.2 | 2.6% |
| red_runners | PPO, 234k steps (xnet3) | 0.050 | 0.508 | 0.03 | 3.6 | 17.4 | 16.3 | 2.2% |
| red_runners | PPO + barrier teacher (phase 1) | 0.000 | 0.025 | 0.00 | 2.7 | 16.7 | 16.3 | 1.1% |
| red_runners | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.000 | 0.00 | 2.5 | 16.7 | 16.2 | 1.2% |
| red_runners | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.025 | 0.00 | 2.4 | 16.3 | 16.3 | 0.9% |
| red_runners | PPO + barrier teacher (phase 1, GPU retrain) | 0.024 | 0.024 | 0.00 | 2.8 | 17.3 | 16.5 | 1.6% |
| red_runners | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.024 | 0.024 | 0.02 | 2.8 | 16.9 | 16.4 | 1.7% |
| red_runners | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.024 | 0.00 | 2.9 | 16.7 | 16.3 | 1.6% |
| inattentive | rule-based AV | 0.000 | 0.000 | 0.03 | 18.4 | 17.2 | 16.3 | 0.0% |
| inattentive | imitation (BC) | 0.000 | 1.847 | 0.18 | 6.7 | 17.4 | 16.3 | 17.7% |
| inattentive | PPO, AVs-only world | 0.025 | 71.542 | 3.61 | 4.1 | 17.5 | 16.1 | 18.4% |
| inattentive | PPO, full world | 0.000 | 78.595 | 3.06 | 4.8 | 17.3 | 16.1 | 19.8% |
| inattentive | PPO, full world, no imitation | 0.051 | 60.642 | 2.52 | 5.2 | 17.3 | 16.1 | 16.2% |
| inattentive | PPO, failure-focused replay | 0.051 | 71.130 | 3.87 | 4.4 | 17.3 | 16.1 | 21.6% |
| inattentive | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.02 | 2.7 | 9.5 | 14.3 | 0.2% |
| inattentive | PPO, 130k steps (xnet) | 0.000 | 2.386 | 0.00 | 2.2 | 16.4 | 16.0 | 2.5% |
| inattentive | PPO, 234k steps (xnet3) | 0.025 | 0.873 | 0.00 | 3.3 | 17.2 | 16.2 | 2.3% |
| inattentive | PPO + barrier teacher (phase 1) | 0.000 | 0.050 | 0.03 | 2.4 | 16.5 | 16.1 | 1.1% |
| inattentive | PPO + barrier teacher, then no filter (phase 2) | 0.025 | 0.100 | 0.00 | 2.5 | 16.5 | 16.1 | 1.2% |
| inattentive | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.050 | 0.00 | 2.1 | 16.1 | 16.1 | 0.9% |
| inattentive | PPO + barrier teacher (phase 1, GPU retrain) | 0.024 | 0.024 | 0.00 | 3.4 | 17.2 | 16.0 | 1.5% |
| inattentive | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.000 | 0.000 | 0.00 | 3.1 | 16.8 | 15.8 | 1.6% |
| inattentive | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.047 | 0.047 | 0.00 | 3.1 | 16.5 | 15.9 | 1.5% |
| jaywalkers | rule-based AV | 0.045 | 0.045 | 0.00 | 11.0 | 12.2 | 11.5 | 0.0% |
| jaywalkers | imitation (BC) | 0.022 | 3.352 | 0.05 | 3.7 | 12.3 | 11.5 | 20.7% |
| jaywalkers | PPO, AVs-only world | 0.045 | 68.528 | 2.29 | 2.4 | 12.3 | 11.4 | 19.1% |
| jaywalkers | PPO, full world | 0.045 | 71.882 | 2.38 | 2.6 | 12.1 | 11.4 | 17.2% |
| jaywalkers | PPO, full world, no imitation | 0.045 | 58.395 | 1.97 | 2.6 | 12.2 | 11.4 | 13.2% |
| jaywalkers | PPO, failure-focused replay | 0.067 | 61.478 | 1.86 | 2.7 | 12.1 | 11.4 | 16.0% |
| jaywalkers | PPO, full world, 10x envelope penalty | 0.000 | 0.022 | 0.00 | 2.3 | 6.8 | 10.3 | 0.2% |
| jaywalkers | PPO, 130k steps (xnet) | 0.045 | 4.243 | 0.05 | 1.3 | 11.9 | 11.3 | 1.9% |
| jaywalkers | PPO, 234k steps (xnet3) | 0.022 | 1.032 | 0.00 | 2.0 | 12.4 | 11.5 | 1.8% |
| jaywalkers | PPO + barrier teacher (phase 1) | 0.022 | 0.245 | 0.00 | 1.4 | 11.9 | 11.5 | 1.0% |
| jaywalkers | PPO + barrier teacher, then no filter (phase 2) | 0.022 | 0.268 | 0.02 | 1.6 | 11.8 | 11.4 | 1.2% |
| jaywalkers | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.022 | 0.089 | 0.02 | 1.5 | 11.5 | 11.4 | 0.8% |
| jaywalkers | PPO + barrier teacher (phase 1, GPU retrain) | 0.022 | 0.379 | 0.02 | 1.7 | 12.5 | 11.8 | 1.7% |
| jaywalkers | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.044 | 0.088 | 0.02 | 1.8 | 12.2 | 11.7 | 1.5% |
| jaywalkers | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.022 | 0.044 | 0.02 | 1.7 | 12.0 | 11.7 | 1.4% |
| blockages | rule-based AV | 0.000 | 0.000 | 0.00 | 12.7 | 14.7 | 10.7 | 0.0% |
| blockages | imitation (BC) | 0.000 | 3.075 | 0.07 | 5.3 | 14.8 | 10.7 | 20.6% |
| blockages | PPO, AVs-only world | 0.000 | 81.283 | 2.13 | 3.2 | 15.1 | 10.7 | 21.5% |
| blockages | PPO, full world | 0.000 | 82.324 | 1.68 | 3.7 | 14.9 | 10.7 | 18.5% |
| blockages | PPO, full world, no imitation | 0.000 | 65.351 | 1.21 | 4.0 | 14.8 | 10.7 | 15.4% |
| blockages | PPO, failure-focused replay | 0.000 | 65.276 | 1.93 | 3.5 | 14.9 | 10.6 | 18.7% |
| blockages | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 1.9 | 8.0 | 9.8 | 0.2% |
| blockages | PPO, 130k steps (xnet) | 0.000 | 3.008 | 0.04 | 2.1 | 14.3 | 10.6 | 2.5% |
| blockages | PPO, 234k steps (xnet3) | 0.000 | 1.482 | 0.00 | 3.0 | 14.9 | 10.7 | 2.8% |
| blockages | PPO + barrier teacher (phase 1) | 0.000 | 0.000 | 0.00 | 1.9 | 14.4 | 10.7 | 1.2% |
| blockages | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.000 | 0.00 | 2.3 | 14.5 | 10.7 | 1.4% |
| blockages | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.034 | 0.00 | 2.1 | 14.1 | 10.6 | 1.0% |
| blockages | PPO + barrier teacher (phase 1, GPU retrain) | 0.032 | 0.032 | 0.00 | 2.7 | 14.5 | 10.4 | 1.7% |
| blockages | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.000 | 0.032 | 0.00 | 2.4 | 14.3 | 10.4 | 1.8% |
| blockages | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.063 | 0.03 | 2.6 | 14.2 | 10.4 | 1.6% |
| av_majority | rule-based AV | 0.000 | 0.000 | 0.00 | 18.9 | 17.4 | 14.6 | 0.0% |
| av_majority | imitation (BC) | 0.000 | 1.983 | 0.21 | 6.9 | 17.5 | 14.7 | 20.5% |
| av_majority | PPO, AVs-only world | 0.016 | 76.484 | 4.66 | 3.9 | 17.6 | 14.5 | 19.9% |
| av_majority | PPO, full world | 0.000 | 85.397 | 4.13 | 4.5 | 17.3 | 14.5 | 19.7% |
| av_majority | PPO, full world, no imitation | 0.000 | 69.528 | 3.71 | 5.1 | 17.3 | 14.4 | 16.4% |
| av_majority | PPO, failure-focused replay | 0.000 | 77.598 | 4.44 | 4.5 | 17.4 | 14.5 | 21.9% |
| av_majority | PPO, full world, 10x envelope penalty | 0.000 | 0.007 | 0.00 | 2.7 | 9.0 | 11.1 | 0.2% |
| av_majority | PPO, 130k steps (xnet) | 0.000 | 3.179 | 0.02 | 2.2 | 16.6 | 14.4 | 2.7% |
| av_majority | PPO, 234k steps (xnet3) | 0.015 | 0.644 | 0.01 | 3.3 | 17.3 | 14.5 | 2.3% |
| av_majority | PPO + barrier teacher (phase 1) | 0.008 | 0.023 | 0.01 | 2.8 | 16.5 | 14.4 | 1.1% |
| av_majority | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.008 | 0.00 | 2.7 | 16.5 | 14.3 | 1.3% |
| av_majority | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.030 | 0.01 | 2.5 | 16.1 | 14.1 | 0.9% |
| av_majority | PPO + barrier teacher (phase 1, GPU retrain) | 0.008 | 0.016 | 0.01 | 3.3 | 17.7 | 14.6 | 1.6% |
| av_majority | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.000 | 0.000 | 0.01 | 3.5 | 17.2 | 14.5 | 1.7% |
| av_majority | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.000 | 0.01 | 3.6 | 16.8 | 14.3 | 1.6% |
| cyclists | rule-based AV | 0.000 | 0.000 | 0.03 | 16.8 | 17.2 | 15.2 | 0.0% |
| cyclists | imitation (BC) | 0.000 | 2.140 | 0.21 | 7.5 | 17.4 | 15.2 | 17.8% |
| cyclists | PPO, AVs-only world | 0.000 | 72.668 | 3.29 | 4.8 | 17.6 | 15.2 | 18.5% |
| cyclists | PPO, full world | 0.000 | 80.962 | 4.04 | 6.1 | 17.2 | 15.1 | 19.2% |
| cyclists | PPO, full world, no imitation | 0.000 | 58.993 | 2.96 | 6.0 | 17.1 | 15.1 | 15.5% |
| cyclists | PPO, failure-focused replay | 0.000 | 70.037 | 3.55 | 5.2 | 17.2 | 15.2 | 20.5% |
| cyclists | PPO, full world, 10x envelope penalty | 0.000 | 0.027 | 0.03 | 2.8 | 9.3 | 13.6 | 0.2% |
| cyclists | PPO, 130k steps (xnet) | 0.028 | 2.058 | 0.03 | 2.1 | 16.4 | 15.1 | 2.5% |
| cyclists | PPO, 234k steps (xnet3) | 0.028 | 0.394 | 0.00 | 3.1 | 17.2 | 15.1 | 2.3% |
| cyclists | PPO + barrier teacher (phase 1) | 0.000 | 0.000 | 0.00 | 2.7 | 16.3 | 15.1 | 1.1% |
| cyclists | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.055 | 0.00 | 2.6 | 16.3 | 15.1 | 1.2% |
| cyclists | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.000 | 0.00 | 2.5 | 16.2 | 15.2 | 0.9% |
| cyclists | PPO + barrier teacher (phase 1, GPU retrain) | 0.000 | 0.027 | 0.00 | 3.0 | 17.1 | 15.4 | 1.6% |
| cyclists | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.027 | 0.027 | 0.00 | 3.0 | 16.7 | 15.3 | 1.7% |
| cyclists | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.027 | 0.055 | 0.05 | 3.5 | 16.4 | 15.2 | 1.5% |
| wet_road | rule-based AV | 0.000 | 0.000 | 0.00 | 19.8 | 17.3 | 16.4 | 0.0% |
| wet_road | imitation (BC) | 0.000 | 1.584 | 0.18 | 7.1 | 17.3 | 16.3 | 17.9% |
| wet_road | PPO, AVs-only world | 0.000 | 70.679 | 4.05 | 4.4 | 17.6 | 16.4 | 18.3% |
| wet_road | PPO, full world | 0.000 | 80.184 | 4.03 | 5.3 | 17.3 | 16.3 | 19.6% |
| wet_road | PPO, full world, no imitation | 0.000 | 61.237 | 4.16 | 5.5 | 17.5 | 16.3 | 16.1% |
| wet_road | PPO, failure-focused replay | 0.000 | 73.447 | 3.63 | 5.6 | 17.3 | 16.3 | 21.4% |
| wet_road | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.05 | 2.9 | 9.6 | 14.6 | 0.2% |
| wet_road | PPO, 130k steps (xnet) | 0.000 | 3.350 | 0.56 | 2.4 | 16.7 | 16.3 | 2.7% |
| wet_road | PPO, 234k steps (xnet3) | 0.000 | 0.459 | 0.00 | 3.8 | 17.4 | 16.5 | 2.2% |
| wet_road | PPO + barrier teacher (phase 1) | 0.000 | 0.025 | 0.00 | 2.9 | 16.6 | 16.4 | 1.1% |
| wet_road | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.025 | 0.00 | 2.7 | 16.5 | 16.3 | 1.2% |
| wet_road | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.000 | 0.00 | 2.4 | 16.2 | 16.2 | 0.9% |
| wet_road | PPO + barrier teacher (phase 1, GPU retrain) | 0.000 | 0.024 | 0.02 | 2.9 | 17.7 | 16.6 | 1.6% |
| wet_road | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.048 | 0.024 | 0.02 | 3.0 | 17.3 | 16.4 | 1.7% |
| wet_road | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.024 | 0.00 | 3.1 | 16.9 | 16.4 | 1.6% |
| squirrel_hill | rule-based AV | 0.000 | 0.000 | 0.00 | 14.3 | 18.8 | 16.7 | 0.0% |
| squirrel_hill | imitation (BC) | 0.000 | 0.890 | 0.19 | 7.6 | 18.9 | 16.6 | 19.0% |
| squirrel_hill | PPO, AVs-only world | 0.000 | 72.815 | 4.01 | 5.3 | 19.2 | 16.7 | 20.7% |
| squirrel_hill | PPO, full world | 0.000 | 83.996 | 4.49 | 6.3 | 19.0 | 16.6 | 22.5% |
| squirrel_hill | PPO, full world, no imitation | 0.000 | 64.579 | 4.41 | 6.4 | 19.0 | 16.7 | 18.8% |
| squirrel_hill | PPO, failure-focused replay | 0.000 | 77.925 | 5.00 | 5.7 | 18.9 | 16.6 | 25.5% |
| squirrel_hill | PPO, full world, 10x envelope penalty | 0.033 | 0.033 | 0.00 | 3.5 | 10.3 | 15.0 | 0.3% |
| squirrel_hill | PPO, 130k steps (xnet) | 0.000 | 2.261 | 0.11 | 3.1 | 18.0 | 16.6 | 3.3% |
| squirrel_hill | PPO, 234k steps (xnet3) | 0.037 | 0.297 | 0.00 | 5.5 | 18.9 | 16.8 | 3.2% |
| squirrel_hill | PPO + barrier teacher (phase 1) | 0.000 | 0.036 | 0.04 | 4.2 | 18.0 | 16.9 | 1.1% |
| squirrel_hill | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.000 | 0.04 | 3.9 | 18.0 | 16.7 | 1.4% |
| squirrel_hill | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.036 | 0.036 | 0.04 | 3.4 | 17.7 | 16.7 | 1.0% |
| squirrel_hill | PPO + barrier teacher (phase 1, GPU retrain) | 0.000 | 0.078 | 0.00 | 4.7 | 18.2 | 16.9 | 1.8% |
| squirrel_hill | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.000 | 0.078 | 0.04 | 4.3 | 17.7 | 16.9 | 2.0% |
| squirrel_hill | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.000 | 0.039 | 0.00 | 4.1 | 17.7 | 17.0 | 1.9% |
| downtown | rule-based AV | 0.000 | 0.000 | 0.00 | 27.0 | 14.9 | 13.5 | 0.0% |
| downtown | imitation (BC) | 0.000 | 2.039 | 0.25 | 14.5 | 14.8 | 13.5 | 21.1% |
| downtown | PPO, AVs-only world | 0.000 | 71.693 | 3.20 | 8.4 | 14.5 | 13.3 | 17.9% |
| downtown | PPO, full world | 0.000 | 78.446 | 3.99 | 9.0 | 14.5 | 13.2 | 18.5% |
| downtown | PPO, full world, no imitation | 0.000 | 65.553 | 3.22 | 9.2 | 14.5 | 13.3 | 15.4% |
| downtown | PPO, failure-focused replay | 0.000 | 66.712 | 4.95 | 9.3 | 14.6 | 13.3 | 18.5% |
| downtown | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 8.1 | 9.9 | 12.2 | 0.4% |
| downtown | PPO, 130k steps (xnet) | 0.000 | 1.681 | 0.06 | 4.7 | 13.9 | 13.2 | 2.9% |
| downtown | PPO, 234k steps (xnet3) | 0.000 | 0.418 | 0.00 | 7.5 | 14.8 | 13.2 | 3.0% |
| downtown | PPO + barrier teacher (phase 1) | 0.000 | 0.116 | 0.00 | 6.8 | 13.9 | 13.2 | 0.9% |
| downtown | PPO + barrier teacher, then no filter (phase 2) | 0.000 | 0.058 | 0.00 | 6.1 | 14.0 | 13.2 | 0.8% |
| downtown | PPO + barrier teacher, speed + jaywalkers (phase 3) | 0.000 | 0.000 | 0.00 | 5.4 | 13.6 | 13.3 | 0.5% |
| downtown | PPO + barrier teacher (phase 1, GPU retrain) | 0.061 | 0.061 | 0.00 | 7.6 | 13.3 | 13.2 | 1.3% |
| downtown | PPO + barrier teacher, then no filter (phase 2, GPU retrain) | 0.060 | 0.000 | 0.00 | 7.6 | 12.9 | 13.2 | 1.1% |
| downtown | PPO + barrier teacher, speed + jaywalkers (phase 3, GPU retrain) | 0.120 | 0.060 | 0.00 | 8.2 | 12.7 | 13.2 | 0.9% |
