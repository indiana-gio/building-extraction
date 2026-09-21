#! /bin/bash
while IFS="," read  -r objectid url
do
  echo $url
done < <(tail $1 -n +2 | tail -n +$2 | head -n $3)