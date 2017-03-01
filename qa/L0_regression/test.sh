#!/bin/bash

THIS_DIR=$(cd $(dirname $0); pwd)

STATUS=0
cd ${THIS_DIR}

for bug in *.py; do
    echo "Running ${bug}..."
    python ${bug}  
    if [ $? == 1 ]; then {
	STATUS=1
	echo 'FAIL'
    } else
	echo 'OK'
    fi 
done

echo "Exit Status = ${STATUS}" 
exit ${STATUS}
