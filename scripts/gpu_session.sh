#!/usr/bin/env bash
# One GPU session for av_world on Google Cloud, from your own machine (needs gcloud logged in and a project set).
#
#   bash scripts/gpu_session.sh                 # evaluate: the latent planner's AVBench runs, the report, the GIFs
#   STAGES="train eval" bash scripts/gpu_session.sh   # also retrain PPO (barrier-teacher phases 1-3), the diffusion world model
#                                                     # and the latent world model first, and re-benchmark them
#
# Cost guard: one g2-standard-8 (1 x NVIDIA L4, ~$0.85/h on demand). The VM is created with --max-run-duration and
# --instance-termination-action=DELETE, so Google deletes it after MAX_MIN minutes even if this script or your
# machine dies; the script also deletes it on exit. At the default 150 minutes that is at most about $2.20, far under $20.
# Results come back into this checkout (results/, docs/img/); review them, then commit and push.
set -euo pipefail
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
# L4s are often out of stock across whole regions, so the list spans continents (on 2026-10-22 only asia-northeast1-a had one)
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b us-west1-c us-west4-a us-west4-c europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west2-a europe-west2-b europe-west3-a europe-west3-b asia-northeast1-a asia-northeast1-b asia-northeast1-c asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-east1-a asia-east1-b asia-east1-c}"
NAME="${NAME:-avworld-gpu-$(date +%H%M%S)}"
MAX_MIN="${MAX_MIN:-150}"
STAGES="${STAGES:-eval}"
[ -n "$PROJECT" ] || { echo "set a project: gcloud config set project YOUR_PROJECT_ID"; exit 1; }
cd "$(git rev-parse --show-toplevel)"

ZONE=""
cleanup() { [ -n "$ZONE" ] && gcloud compute instances delete "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet || true; }
trap cleanup EXIT

for z in $ZONES; do  # L4s are often out of stock in one zone: try the next
  if gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$z" --machine-type g2-standard-8 \
      --image-family common-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
      --boot-disk-size 100GB --maintenance-policy TERMINATE --provisioning-model STANDARD \
      --max-run-duration "${MAX_MIN}m" --instance-termination-action DELETE \
      --metadata install-nvidia-driver=True --quiet; then ZONE="$z"; break; fi
done
[ -n "$ZONE" ] || { echo "no zone had an L4 free"; exit 1; }
echo "VM $NAME in $ZONE (deletes itself after $MAX_MIN min)"
SSH=(gcloud compute ssh "$NAME" --zone "$ZONE" --project "$PROJECT" --quiet --command)
for i in $(seq 1 30); do "${SSH[@]}" "nvidia-smi -L" 2>/dev/null && break; sleep 10; done

git ls-files -z | tar --null -T - -czf /tmp/avworld_src.tgz  # the tracked tree, including your local edits
gcloud compute scp /tmp/avworld_src.tgz "$NAME:~/src.tgz" --zone "$ZONE" --project "$PROJECT" --quiet

"${SSH[@]}" "set -e; mkdir -p av_world && tar -xzf src.tgz -C av_world && cd av_world
sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update >/dev/null && sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3.10-venv >/dev/null  # not on the image
python3 -m venv ~/venv && . ~/venv/bin/activate && pip -q install --upgrade pip && pip -q install -e '.[gpu,dev]'
python -c 'import jax; print(jax.devices())'
export PYTHONPATH=.
if echo '$STAGES' | grep -q train; then
  for k in 1 2 3; do python -m avrl.unaided \$k; done                      # PPO with the barrier teacher, phases 1-3
  # the diffusion world model: results/dwm/final trains on small_av_world logs (data/dwm, not in this repo), so retrain the
  # self-contained variant, on logs from the av_world area simulator, instead
  python -m dwm.xnet_data results/xnet2/policy.npz && DWM_SOURCE=xnet python -m dwm.train --data data/dwm_xnet --budget 240 --out results/dwm/xnet
  python -m wmplan.data && python -m wmplan.model                           # the latent world model
  python -m avrl.bench forget 'PPO + barrier teacher (phase 1)' 'PPO + barrier teacher, then no filter (phase 2)' \
    'PPO + barrier teacher, speed + jaywalkers (phase 3)' 'latent world-model planner (wm-mpc)'
  python -m avrl.bench run                                                  # the retrained policies, every scenario
fi
if echo '$STAGES' | grep -q eval; then
  python -m avrl.bench run-wm                                               # the latent planner: 3 scenarios x 2 modes
  python -m avrl.bench report && python -m avrl.bench plot
  python -m wmplan.viz                                                      # planner GIFs at junctions and merges
fi
tar -czf ~/out.tgz results docs/img"
gcloud compute scp "$NAME:~/out.tgz" /tmp/avworld_out.tgz --zone "$ZONE" --project "$PROJECT" --quiet
tar -xzf /tmp/avworld_out.tgz
echo "done: results are in results/ and docs/img/ (git status to review). The VM is deleted next."
