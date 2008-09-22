#!/bin/bash

mkdir -p html/api
epydoc --config local.epydoc

cd doc
sh build_html.sh
cd ../
rm -Rf html/doc
mv doc/html html/doc

