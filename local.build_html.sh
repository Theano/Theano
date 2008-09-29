#!/bin/bash

mkdir -p html/api && mkdir -p html/doc

# this builds some stuff or something... basically makes the rest work properly
# for a reason I don't understand.  -JB 20080924
python __init__.py

#runs if you called $./local.build_html.sh epydoc
if [ " $1" != " rst" ]; then
./epydoc --config local.epydoc
fi

#runs if you called $./local.build_html.sh rst
if [ " $1" != " epydoc" ]; then
    APIRST2HTML=doc/apirst2html.py
    EPYDOC_ARGS='--external-api=api --external-api-file=api:html/api/api-objects.txt --external-api-root=api:../api/ --link-stylesheet'

    # install the stylesheets
    HTML4CSS1='/usr/lib/python2.5/site-packages/docutils/writers/html4css1/html4css1.css'
    cp $HTML4CSS1 html/html4css1.css
    cp doc/colorful.css html/colorful.css
    cp doc/style.css html/style.css

    #generate the index & readme files
    echo "$APIRST2HTML $EPYDOC_ARGS index.txt html/index.html..."
    $APIRST2HTML -stg $EPYDOC_ARGS --stylesheet=style.css index.txt html/index.html
    echo "$APIRST2HTML $EPYDOC_ARGS README.txt html/README.html..."
    $APIRST2HTML -stg $EPYDOC_ARGS --stylesheet=style.css README.txt html/README.html

    #generate the oplist in ReST format
    echo "gen oplist..."
    python gen_oplist.py > doc/oplist.txt
    python gen_typelist.py > doc/typelist.txt

    #generate html files for all the ReST documents in doc/
    echo "gen doc/*.txt..."
    for RST in doc/*.txt;  do
        BASENAME=$(basename $RST .txt)
        echo "gen doc/$BASENAME.txt..."
        $APIRST2HTML -stg $EPYDOC_ARGS --stylesheet=../style.css doc/$BASENAME.txt html/doc/$BASENAME.html
    done
fi

