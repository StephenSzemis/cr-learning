CORPUS=../data/tokens.txt
VOCAB_FILE=../data/vocab.txt

COOCCURRENCE_FILE=../data/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=../data/cooccurrence.shuf.bin

BUILDDIR=../../glove/build

SAVE_FILE=../data/vectors

VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
VOCAB_MAX_COUNT=100000
VECTOR_SIZE=60
MAX_ITER=12
WINDOW_SIZE=20
BINARY=2
NUM_THREADS=8
X_MAX=10

$BUILDDIR/vocab_count -max-vocab $VOCAB_MAX_COUNT -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE
