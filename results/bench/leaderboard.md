| policy | checks passed (of 72) | scenarios overcome (of 9) | nominal | rush_hour | red_runners | inattentive | jaywalkers | blockages | av_majority | squirrel_hill | downtown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule-based AV | 72 | 9 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| PPO, 234k steps (xnet3) | 55 | 0 | 6/8 | 6/8 | 6/8 | 6/8 | 5/8 | 7/8 | 6/8 | 7/8 | 6/8 |
| PPO, full world, 10x envelope penalty | 52 | 0 | 5/8 | 6/8 | 6/8 | 5/8 | 6/8 | 6/8 | 6/8 | 6/8 | 6/8 |
| imitation (BC) | 46 | 0 | 5/8 | 5/8 | 5/8 | 5/8 | 6/8 | 4/8 | 5/8 | 5/8 | 6/8 |
| PPO, AVs-only world | 43 | 0 | 5/8 | 5/8 | 5/8 | 5/8 | 4/8 | 5/8 | 5/8 | 4/8 | 5/8 |
| PPO, 130k steps (xnet) | 43 | 0 | 6/8 | 4/8 | 4/8 | 6/8 | 3/8 | 6/8 | 4/8 | 6/8 | 4/8 |
| PPO, failure-focused replay | 41 | 0 | 5/8 | 5/8 | 4/8 | 5/8 | 3/8 | 5/8 | 5/8 | 4/8 | 5/8 |
| PPO, full world | 39 | 0 | 5/8 | 5/8 | 4/8 | 5/8 | 3/8 | 4/8 | 4/8 | 4/8 | 5/8 |
| PPO, full world, no imitation | 38 | 0 | 5/8 | 5/8 | 4/8 | 5/8 | 3/8 | 5/8 | 3/8 | 4/8 | 4/8 |

| scenario | policy | crashes /AV-h | raw crashes /AV-h | raw near misses /AV-h | hard brakes /AV-h | AV km/h | human km/h | envelope cuts |
|---|---|---|---|---|---|---|---|---|
| nominal | rule-based AV | 0.023 | 0.023 | 0.00 | 17.4 | 16.8 | 16.2 | 0.0% |
| nominal | imitation (BC) | 0.000 | 2.119 | 0.12 | 6.1 | 17.0 | 16.2 | 18.2% |
| nominal | PPO, AVs-only world | 0.000 | 48.235 | 0.11 | 3.8 | 17.3 | 16.2 | 19.2% |
| nominal | PPO, full world | 0.000 | 52.227 | 0.17 | 5.0 | 17.1 | 16.2 | 19.5% |
| nominal | PPO, full world, no imitation | 0.000 | 42.818 | 0.20 | 5.0 | 17.2 | 16.2 | 16.2% |
| nominal | PPO, failure-focused replay | 0.000 | 44.875 | 0.16 | 4.7 | 17.1 | 16.2 | 21.7% |
| nominal | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.02 | 2.7 | 9.5 | 14.2 | 0.3% |
| nominal | PPO, 130k steps (xnet) | 0.000 | 2.695 | 0.05 | 2.0 | 16.6 | 16.1 | 2.7% |
| nominal | PPO, 234k steps (xnet3) | 0.000 | 0.419 | 0.02 | 3.3 | 17.2 | 16.3 | 2.2% |
| rush_hour | rule-based AV | 0.016 | 0.016 | 0.00 | 16.4 | 14.6 | 13.9 | 0.0% |
| rush_hour | imitation (BC) | 0.000 | 1.863 | 0.18 | 5.5 | 14.8 | 13.9 | 18.5% |
| rush_hour | PPO, AVs-only world | 0.000 | 56.826 | 0.42 | 3.6 | 14.6 | 13.9 | 17.6% |
| rush_hour | PPO, full world | 0.000 | 60.139 | 0.31 | 4.0 | 14.6 | 13.9 | 16.8% |
| rush_hour | PPO, full world, no imitation | 0.000 | 46.570 | 0.48 | 4.5 | 14.6 | 13.9 | 13.3% |
| rush_hour | PPO, failure-focused replay | 0.000 | 49.289 | 0.19 | 4.4 | 14.6 | 13.9 | 17.8% |
| rush_hour | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 2.9 | 8.3 | 11.9 | 0.2% |
| rush_hour | PPO, 130k steps (xnet) | 0.000 | 2.376 | 0.02 | 1.9 | 14.0 | 13.7 | 2.3% |
| rush_hour | PPO, 234k steps (xnet3) | 0.000 | 0.629 | 0.03 | 3.0 | 14.6 | 13.9 | 2.0% |
| red_runners | rule-based AV | 0.047 | 0.047 | 0.05 | 19.0 | 17.5 | 16.7 | 0.0% |
| red_runners | imitation (BC) | 0.000 | 2.031 | 0.07 | 5.7 | 17.7 | 16.7 | 18.5% |
| red_runners | PPO, AVs-only world | 0.000 | 50.030 | 0.11 | 4.0 | 17.7 | 16.6 | 19.4% |
| red_runners | PPO, full world | 0.023 | 55.008 | 0.18 | 4.7 | 17.4 | 16.5 | 20.0% |
| red_runners | PPO, full world, no imitation | 0.000 | 43.992 | 0.21 | 5.0 | 17.5 | 16.4 | 16.4% |
| red_runners | PPO, failure-focused replay | 0.000 | 48.003 | 0.22 | 4.6 | 17.6 | 16.5 | 22.4% |
| red_runners | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 2.9 | 9.7 | 14.6 | 0.3% |
| red_runners | PPO, 130k steps (xnet) | 0.023 | 2.505 | 0.07 | 2.3 | 16.9 | 16.5 | 2.7% |
| red_runners | PPO, 234k steps (xnet3) | 0.000 | 0.446 | 0.07 | 3.5 | 17.8 | 16.7 | 2.3% |
| inattentive | rule-based AV | 0.069 | 0.069 | 0.00 | 18.1 | 16.6 | 15.8 | 0.0% |
| inattentive | imitation (BC) | 0.023 | 2.081 | 0.10 | 6.2 | 16.7 | 15.8 | 18.6% |
| inattentive | PPO, AVs-only world | 0.046 | 48.836 | 0.11 | 4.0 | 17.0 | 15.8 | 19.6% |
| inattentive | PPO, full world | 0.023 | 53.106 | 0.23 | 4.8 | 16.8 | 15.8 | 19.4% |
| inattentive | PPO, full world, no imitation | 0.000 | 43.390 | 0.25 | 5.0 | 16.9 | 15.8 | 16.1% |
| inattentive | PPO, failure-focused replay | 0.000 | 45.416 | 0.26 | 4.6 | 16.9 | 15.9 | 21.8% |
| inattentive | PPO, full world, 10x envelope penalty | 0.022 | 0.000 | 0.02 | 3.0 | 9.4 | 13.9 | 0.4% |
| inattentive | PPO, 130k steps (xnet) | 0.023 | 2.705 | 0.05 | 2.1 | 16.3 | 15.8 | 2.8% |
| inattentive | PPO, 234k steps (xnet3) | 0.023 | 0.490 | 0.02 | 3.4 | 17.0 | 16.0 | 2.3% |
| jaywalkers | rule-based AV | 0.021 | 0.021 | 0.00 | 12.9 | 12.4 | 11.7 | 0.0% |
| jaywalkers | imitation (BC) | 0.021 | 2.526 | 0.00 | 4.3 | 12.3 | 11.7 | 20.5% |
| jaywalkers | PPO, AVs-only world | 0.021 | 49.466 | 0.15 | 2.6 | 12.4 | 11.7 | 19.1% |
| jaywalkers | PPO, full world | 0.021 | 51.015 | 0.21 | 3.1 | 12.1 | 11.5 | 17.6% |
| jaywalkers | PPO, full world, no imitation | 0.042 | 42.302 | 0.18 | 3.0 | 12.2 | 11.6 | 13.5% |
| jaywalkers | PPO, failure-focused replay | 0.021 | 43.700 | 0.14 | 2.9 | 12.2 | 11.6 | 16.0% |
| jaywalkers | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 2.1 | 7.0 | 10.3 | 0.1% |
| jaywalkers | PPO, 130k steps (xnet) | 0.021 | 3.976 | 0.02 | 1.3 | 11.7 | 11.5 | 2.1% |
| jaywalkers | PPO, 234k steps (xnet3) | 0.021 | 0.758 | 0.02 | 2.7 | 12.4 | 11.7 | 1.8% |
| blockages | rule-based AV | 0.000 | 0.000 | 0.00 | 14.3 | 14.9 | 10.8 | 0.0% |
| blockages | imitation (BC) | 0.092 | 2.328 | 0.10 | 5.4 | 14.9 | 10.8 | 22.5% |
| blockages | PPO, AVs-only world | 0.000 | 54.552 | 0.16 | 3.7 | 15.4 | 10.8 | 21.9% |
| blockages | PPO, full world | 0.031 | 58.475 | 0.33 | 4.0 | 15.1 | 10.8 | 18.1% |
| blockages | PPO, full world, no imitation | 0.000 | 45.291 | 0.21 | 3.4 | 15.0 | 10.8 | 14.8% |
| blockages | PPO, failure-focused replay | 0.000 | 42.484 | 0.20 | 3.9 | 15.0 | 10.9 | 18.7% |
| blockages | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 2.5 | 8.7 | 10.0 | 0.2% |
| blockages | PPO, 130k steps (xnet) | 0.061 | 2.645 | 0.00 | 2.1 | 14.5 | 10.8 | 2.5% |
| blockages | PPO, 234k steps (xnet3) | 0.000 | 1.131 | 0.00 | 2.8 | 15.2 | 10.9 | 2.3% |
| av_majority | rule-based AV | 0.008 | 0.008 | 0.01 | 19.3 | 17.9 | 15.0 | 0.0% |
| av_majority | imitation (BC) | 0.000 | 1.807 | 0.10 | 6.8 | 18.0 | 15.1 | 19.4% |
| av_majority | PPO, AVs-only world | 0.000 | 57.389 | 0.32 | 3.8 | 18.2 | 14.9 | 20.1% |
| av_majority | PPO, full world | 0.000 | 63.088 | 0.34 | 4.6 | 17.8 | 14.8 | 20.6% |
| av_majority | PPO, full world, no imitation | 0.015 | 50.120 | 0.34 | 4.7 | 17.8 | 14.9 | 17.2% |
| av_majority | PPO, failure-focused replay | 0.000 | 54.458 | 0.27 | 4.4 | 17.9 | 15.0 | 22.9% |
| av_majority | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.01 | 2.7 | 9.3 | 11.0 | 0.2% |
| av_majority | PPO, 130k steps (xnet) | 0.000 | 3.163 | 0.03 | 2.4 | 17.0 | 14.6 | 2.9% |
| av_majority | PPO, 234k steps (xnet3) | 0.007 | 0.783 | 0.02 | 3.3 | 18.1 | 15.1 | 2.5% |
| squirrel_hill | rule-based AV | 0.037 | 0.037 | 0.00 | 14.9 | 18.1 | 16.5 | 0.0% |
| squirrel_hill | imitation (BC) | 0.000 | 0.699 | 0.27 | 7.5 | 18.4 | 16.6 | 17.6% |
| squirrel_hill | PPO, AVs-only world | 0.037 | 48.203 | 0.68 | 5.6 | 18.9 | 16.6 | 19.7% |
| squirrel_hill | PPO, full world | 0.037 | 53.147 | 0.81 | 5.6 | 18.5 | 16.7 | 21.7% |
| squirrel_hill | PPO, full world, no imitation | 0.037 | 41.238 | 0.76 | 5.8 | 18.5 | 16.5 | 17.7% |
| squirrel_hill | PPO, failure-focused replay | 0.037 | 48.469 | 0.51 | 5.9 | 18.5 | 16.6 | 24.8% |
| squirrel_hill | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 3.7 | 10.2 | 14.5 | 0.3% |
| squirrel_hill | PPO, 130k steps (xnet) | 0.036 | 2.070 | 0.04 | 3.3 | 17.9 | 16.5 | 3.6% |
| squirrel_hill | PPO, 234k steps (xnet3) | 0.000 | 0.187 | 0.00 | 5.6 | 18.8 | 16.7 | 2.9% |
| downtown | rule-based AV | 0.052 | 0.052 | 0.00 | 25.6 | 14.6 | 13.3 | 0.0% |
| downtown | imitation (BC) | 0.052 | 1.964 | 0.00 | 13.8 | 14.5 | 13.4 | 21.5% |
| downtown | PPO, AVs-only world | 0.052 | 44.808 | 0.71 | 7.7 | 14.6 | 13.3 | 18.9% |
| downtown | PPO, full world | 0.052 | 47.817 | 0.60 | 9.5 | 14.3 | 13.2 | 18.6% |
| downtown | PPO, full world, no imitation | 0.052 | 38.577 | 0.42 | 8.9 | 14.2 | 13.2 | 15.2% |
| downtown | PPO, failure-focused replay | 0.052 | 42.586 | 0.57 | 8.4 | 14.4 | 13.3 | 18.6% |
| downtown | PPO, full world, 10x envelope penalty | 0.000 | 0.000 | 0.00 | 8.1 | 9.9 | 12.0 | 0.6% |
| downtown | PPO, 130k steps (xnet) | 0.000 | 1.226 | 0.05 | 4.9 | 13.7 | 13.1 | 3.2% |
| downtown | PPO, 234k steps (xnet3) | 0.000 | 0.264 | 0.11 | 6.7 | 14.4 | 13.3 | 3.1% |
