conda activate gemini-env

export PARLAY_NUM_THREADS=$(nproc)

for id in coci14c1p4 coci14c2p5 cco08p4 joi13op2 cco16p1 coi06p2 coci08c6p5 coci13c2p5 coci18c1p5 cco08p2 ioi08p2 joi20scd4p1 coci21c3p2 coci21c5p3 coi16p2 coci22c1p3 coci22c3p3 coci23c4p5 ccc24s4
do
    echo "Running openevolve with id=$id"
    python local_run_openevolve.py "$id"
done
