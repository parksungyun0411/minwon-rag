# minwon-rag Kaggle T4 실행 (2026-10-27, docs/kaggle_t4.md). 로컬에서 만든 시점 분할 결과(코퍼스·골든셋)로 색인과 평가만 돈다.
#   index: KURE-v1·bge-m3 임베딩 + kiwi BM25 / eval: configs/matrix.yaml 7개 설정 × 실제 질의 150건
# 저장소는 고정 커밋, 패키지는 uv.lock 그대로(--frozen). 모델 리비전은 logs/models.txt 에 남긴다.
import subprocess

SH = r"""
set -u
W=/kaggle/working; mkdir -p $W/logs $W/out
export HF_HOME=/tmp/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $W/logs/steps.log; }
IN=$(dirname "$(find /kaggle/input -name corpus.jsonl | head -1)")
[ -f "$IN/corpus.jsonl" ] || { log "INPUT_MISSING"; find /kaggle/input -maxdepth 4 | head -50; exit 1; }
git clone -q https://github.com/parksungyun0411/minwon-rag /tmp/mr && cd /tmp/mr && git checkout -q __SHA__ || { log "CLONE_FAILED"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv | tee $W/logs/gpu.txt
uv sync -q --frozen --python 3.12 --extra models > $W/logs/setup.log 2>&1 || { log "SETUP_FAILED"; tail -30 $W/logs/setup.log; exit 1; }
uv pip freeze --python .venv/bin/python > $W/logs/freeze.txt
mkdir -p data/processed data/golden/real results
for z in $(find /kaggle/input -name "*.zip"); do unzip -oq "$z" -d /tmp/in_unzip; done  # 폴더를 zip 으로 둔 경우
Q=$(find /kaggle/input /tmp/in_unzip -name queries.jsonl 2>/dev/null | head -1)
[ -n "$Q" ] || { log "GOLDEN_MISSING"; find /kaggle/input -maxdepth 4; exit 1; }
cp "$IN/corpus.jsonl" data/processed/ && cp "$(dirname "$Q")"/queries.jsonl "$(dirname "$Q")"/qrels.jsonl data/golden/real/ \
  && cp "$IN/split_stats.json" results/ || { log "COPY_FAILED"; exit 1; }
wc -l data/processed/corpus.jsonl data/golden/real/*.jsonl | tee -a $W/logs/steps.log
.venv/bin/python - > $W/logs/models.txt 2>&1 <<'PY'
import torch, yaml
from huggingface_hub import HfApi
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
for alias, mid in yaml.safe_load(open("configs/matrix.yaml"))["models"].items():
    print(alias, mid, HfApi().model_info(mid).sha)
PY
log "SETUP_DONE sha=$(git rev-parse --short HEAD) $(head -1 $W/logs/models.txt)"

pack() {
  ( cd /tmp/mr
    [ -d results ] && mkdir -p $W/out/results && cp -r results/. $W/out/results/
    [ -d data/index ] && tar czf $W/out/index.tgz -C data index )
  return 0
}
step() {  # step <이름> <명령...>
  local name=$1; shift
  log "START $name"
  local t0=$(date +%s)
  "$@" > "$W/logs/$name.log" 2>&1; local rc=$?
  local dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ]; then log "OK $name ${dt}s"; else log "FAILED $name rc=$rc ${dt}s"; tail -20 "$W/logs/$name.log"; fi
  pack; return $rc
}
T=""; [ -x /usr/bin/time ] && T="/usr/bin/time -v"  # 최대 메모리 기록(없으면 생략)
step index $T .venv/bin/minwon index || exit 1
step eval $T .venv/bin/minwon eval || exit 1
grep -h "Maximum resident\|Elapsed (wall" $W/logs/index.log $W/logs/eval.log | tee -a $W/logs/steps.log
du -sh $W/out/* | tee -a $W/logs/steps.log
log "ALL_DONE"
"""
SH = SH.replace("__SHA__", "1ee2d2490aeb7838a0051b91d5fb511ef94bd28b")
subprocess.run(["bash", "-c", SH])
print(open("/kaggle/working/logs/steps.log").read())
