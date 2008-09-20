#!/bin/bash

APIRST2HTML=~/bin/apirst2html.py

EPYDOC_ARGS='--external-api=api --external-api-file=api:../html/api/api-objects.txt --external-api-root=api:epydoc/'

mkdir html 2> /dev/null

for RST in graph ;  do
    $APIRST2HTML $EPYDOC_ARGS $RST.txt html/$RST.html
done
