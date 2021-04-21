#!/bin/bash

make clean && make && make clean_buildfiles


if verapdf -p veraPDF-UK-custom-profile-7987-version1-custom8.xml thesis.pdf | grep -q isCompliant=\"true\"; then
    echo "PDF is compliant with UK PDF/A requirements"
else
    echo "ERROR in PDF/A validation"
fi
