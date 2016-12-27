#!/bin/sh

DATE=`date +%Y%m%d`
LOG_FILE=./regression_result_$DATE.log

if [ -f "$LOG_FILE" ]; then
    rm $LOG_FILE
fi

################################
# Check the coding style
################################
PY_FILE_DIR=../
PY_FILES=`find $PY_FILE_DIR -maxdepth 1 -name "*.py"`

echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "Start coding style check..." 2>&1 | tee -a $LOG_FILE
which flake8 >/dev/null 2>&1
FLAKE8=$?
if [ "$FLAKE8" -eq 0 ]; then
    for file in $PY_FILES
    do
        echo "Running flake8 on $file" 2>&1 | tee -a $LOG_FILE
        flake8 $file  2>&1 | tee -a $LOG_FILE       
        echo "" 2>&1 | tee -a $LOG_FILE
    done
else
    echo "Skip coding style check since flake8 is noe present" 2>&1 | tee -a $LOG_FILE
fi
echo "Coding style check done" 2>&1 | tee -a $LOG_FILE
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE


################################
# Graph generating validation
################################
python -c "import pydot; print pydot.find_graphviz()" >  /dev/null 2>&1
graphSupportted=$?
if [ "$graphSupportted" -eq 0 ]; then
    echo "##################################################" 2>&1 | tee -a $LOG_FILE
    echo "Start graph generating validation..." 2>&1 | tee -a $LOG_FILE
    python gen_combination_graph.py 2>&1 | tee -a $LOG_FILE
    python gen_lrn_graph.py 2>&1 | tee -a $LOG_FILE
    python gen_pool_graph.py 2>&1 | tee -a $LOG_FILE
    python gen_relu_graph.py 2>&1 | tee -a $LOG_FILE
    echo "Graph generating validation done" 2>&1 | tee -a $LOG_FILE
    echo "##################################################" 2>&1 | tee -a $LOG_FILE
    echo "" 2>&1 | tee -a $LOG_FILE
else
    echo "Skip graph generation validation since graph is not supportted on your machine" 2>&1 | tee -a $LOG_FILE
fi


################################
# Unit Test
################################
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "Start unit test..." 2>&1 | tee -a $LOG_FILE
python test_mkl_dummy.py 2>&1 | tee -a $LOG_FILE

# opt
echo "run test_opt.py..." 2>&1 | tee -a $LOG_FILE
python test_opt.py 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# Pooling
echo "run test_pool.py..." 2>&1 | tee -a $LOG_FILE
nosetests -s test_pool.py 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# Relu
echo "run test_relu.py..." 2>&1 | tee -a $LOG_FILE
nosetests -s test_relu.py
echo "" 2>&1 | tee -a $LOG_FILE

# LRN
echo "run test_lrn.py..." 2>&1 | tee -a $LOG_FILE
python test_lrn.py 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# Conv
echo "run test_conv.py..." 2>&1 | tee -a $LOG_FILE
python test_conv.py 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# BN
echo "run test_bn.py..." 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# Elemwise
echo "run test_elemwise.py..." 2>&1 | tee -a $LOG_FILE
python test_elemwise.py 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

# others...
echo "Unit test done" 2>&1 | tee -a $LOG_FILE
echo "##################################################" 2>&1 | tee -a $LOG_FILE
echo "" 2>&1 | tee -a $LOG_FILE

################################
# Clean up
################################
echo "Clean up..." 2>&1 | tee -a $LOG_FILE
rm -rf *.png
