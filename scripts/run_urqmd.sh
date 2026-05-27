#!/bin/bash
# run_urqmd.sh — runs one UrQMD job for a given energy, seed, and output directory
# Arguments: $1=energy tag (e.g. 3p2), $2=seed, $3=output base directory

set -e

ENERGY=$1
SEED=$2
OUTDIR=$3

URQMD_DIR=/star/data03/scratch/apellotji/urqmd-4.0
WORKDIR=$OUTDIR/$ENERGY/job_$SEED

mkdir -p $WORKDIR
mkdir -p $OUTDIR/$ENERGY
cd $WORKDIR

# Energy tag to elb (lab kinetic energy per nucleon in GeV)
case $ENERGY in
    3p2) ELB=4.03 ;;
    3p5) ELB=5.16 ;;
    3p9) ELB=6.87 ;;
    4p5) ELB=10.07 ;;
    *) echo "Unknown energy $ENERGY"; exit 1 ;;
esac

# Write the inputfile for this job
cat > inputfile << EOF
pro 208 82
tar 208 82

nev 5000
IMP 0. 14.

elb $ELB
tim 200 200
rsd $SEED

f13
f15
f16
f19

xxx
EOF

# Copy the executable and required files
cp $URQMD_DIR/urqmd.* $WORKDIR/ 2>/dev/null || true
cp -r $URQMD_DIR/eosfiles $WORKDIR/ 2>/dev/null || true
cp $URQMD_DIR/tables.dat $WORKDIR/ 2>/dev/null || true

# Run UrQMD (reads from inputfile via ftn09)
export ftn09=inputfile
export ftn13=$OUTDIR/$ENERGY/output_$SEED.f13   # flat in energy dir, seed in name
export ftn14=/dev/null
export ftn15=output.f15
export ftn16=output.f16
export ftn19=output.f19
export ftn20=/dev/null

# Find the right executable
EXENAME=$WORKDIR/urqmd.x86_64
if [ ! -f "$EXENAME" ]; then
    echo "No urqmd executable found at $EXENAME"
    exit 1
fi

echo "Running UrQMD: energy=$ENERGY elb=$ELB seed=$SEED nev=5000"
time $EXENAME

echo "Done. f13 written to $OUTDIR/$ENERGY/output_$SEED.f13"
