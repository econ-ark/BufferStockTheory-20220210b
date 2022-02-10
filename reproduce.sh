#!/bin/bash
# Reproduce results then text of the paper 
scriptDir="$(dirname "$0")"

# Regenerate computed results (figs) needed for compiling paper
./reproduce/computed.sh MIN # Replace with MAX to execute ApndxBalancedGrowthcNrmAndCov.ipynb

echo '' ; echo 'Reproduce text of paper:' ; echo ''

texname=BufferStockTheory
output_directory='LaTeX'

# Make figures that get made by executing a latex file
# (they should have a filename ending in Make.tex)
cd Figures
# For this paper, only the tikz figures need to be made by pdflatex - others are made by python
for fName_tikzMake in *Make.tex; do # names of all files ending in Make.tex
    echo "Processing figure $fName_tikzMake"
    fName=${fName_tikzMake%_tikzMake.tex} # Remove the '_tikzMake.tex' part of the filename
    cmd="pdflatex -halt-on-error --output-format pdf -output-directory=../$output_directory $fName_tikzMake"
    echo "$cmd"
    eval "$cmd"
    mv -f                                                             "../$output_directory/$fName" "$fName.pdf"
done
cd ..

# Compile LaTeX files in root directory
for file in "$texname" "$texname"-NoAppendix "$texname"-Slides; do
    echo '' ; echo "Compiling $file" ; echo ''
    cmd="pdflatex -halt-on-error -output-directory=$output_directory $file"
    eval "$cmd"
    eval "$cmd > /dev/null" # Hide second output to reduce clutter
    bibtex $output_directory/"$file"
    eval "$cmd" # Hide third output to reduce clutter
    eval "$cmd > /dev/null" 
    echo '' ; echo "Compiled $file" ; echo ''
done

# Compile All-Figures and All-Tables
for type in Figures Tables; do
    cmd="pdflatex -halt-on-error -output-directory=$output_directory $type/All-$type"
    echo "$cmd" ; eval "$cmd"
    # If there is a .bib file, make the references
    [[ -e "../$output_directory/$type/All-$type.aux" ]] && bibtex "$type/All-$type.bib" && eval "$cmd" && eval "$cmd" 
    mv -f "$output_directory/All-$type.pdf" "$type"  # Move from the LaTeX output directory to the destination
done

# All the appendices can be compiled as standalone documents (they are "subfiles")
# Make a list of all the appendices, put the list in the file /tmp/appendices
find ./Appendices -name '*.tex' ! -name '*econtexRoot*' ! -name '*econtexPath*' -maxdepth 1 -exec echo {} \; > /tmp/appendices

# For each appendix process it by pdflatex
# If it contains a standalone bibliography, process that
# Then rerun pdflatex to complete the processing and move the resulting pdf file

while read appendixName; do
    filename=$(basename ${appendixName%.*}) # Strip the path and the ".tex"
    cmd="pdflatex -halt-on-error                 --output-directory=$output_directory $appendixName"
    echo "$cmd"
    eval "$cmd"
    if grep -q 'bibliography{' "$appendixName"; then # it has a bibliography
	bibtex $output_directory/$filename 
	eval "$cmd" 
    fi
    eval "$cmd"
    cmd="mv $output_directory/$filename.pdf Appendices"
    echo "$cmd"
    eval "$cmd"
done < /tmp/appendices

[[ -e "$texname".pdf ]] && rm -f "$texname".pdf

echo '' 

if [[ -e "$output_directory/$texname.pdf" ]]; then
    echo "Paper has been compiled to $output_directory/$texname.pdf"
    echo "and copied to ./$texname.pdf"
    cp "$output_directory/$texname.pdf" "./$texname.pdf"
else
    echo "Something went wrong and the paper is not in $output_directory/$texname.pdf"
fi

echo ''

