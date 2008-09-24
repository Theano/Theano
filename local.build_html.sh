#!/bin/bash

APIRST2HTML=doc/apirst2html.py
EPYDOC_ARGS='--external-api=api --external-api-file=api:html/api/api-objects.txt --external-api-root=api:../api/'


mkdir -p html/api && mkdir -p html/doc

# this builds some stuff or something... basically makes the rest work properly
# for a reason I don't understand.  -JB 20080924
python __init__.py

if [ " $1" != " rst" ]; then
epydoc --config local.epydoc
fi

if [ " $1" != " epydoc" ]; then
python gen_oplist.py > doc/oplist.txt
for RST in graph oplist ;  do

    $APIRST2HTML $EPYDOC_ARGS doc/$RST.txt html/doc/$RST.html
done
fi

