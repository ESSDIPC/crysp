#!/usr/bin/env bash

FILES="$1/*.csv"
READ_FIRST=0
for f in $FILES
do
  if  [ $READ_FIRST -eq 0 ]
  then
    LINES_NUMBER=`wc -l < $f`
    READ_FIRST=1
    # store number of time ticks per event in the first line
    echo "TICKS," $((LINES_NUMBER-16)) > $1.csv
  fi

  echo "Processing $f file..."
  # skip header for each file
  tail -n +17 "$f" >> $1.csv
done

sed -i '' -e 's/inf/99999999/g' $1.csv
zip $1.zip $1.csv
rm $1.csv
