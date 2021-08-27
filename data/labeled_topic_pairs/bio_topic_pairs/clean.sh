# remove texts of topics' documents from the annotation files

outfold='cleaned'
pattern='MATCH\:\|WORDS\:\|PAIR_ID\:\|TID\:\|\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-'
for filename in *.txt; do 
    grep $pattern $filename > "$outfold/$filename"
    echo "$outfold/$filename"
done

