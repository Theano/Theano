#!/bin/bash

APIRST2HTML=doc/apirst2html.py
EPYDOC_ARGS='--external-api=api --external-api-file=api:html/api/api-objects.txt --external-api-root=api:../api/'


mkdir -p html/api 
mkdir -p html/doc

if [ " $1" != " rst" ]; then
epydoc --config local.epydoc
fi

python gen_op_list.py > doc/oplist.txt

if [ " $1" != " epydoc" ]; then
for RST in graph oplist ;  do
    $APIRST2HTML $EPYDOC_ARGS doc/$RST.txt html/doc/$RST.html
done
fi

