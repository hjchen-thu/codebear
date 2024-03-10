#!/bin/bash

f_value=""
q_value=""

while getopts ":f:q:" opt; do
  case ${opt} in
    f )
      f_value=$OPTARG
      ;;
    q )
      q_value=$OPTARG
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      exit 1
      ;;
  esac
done
shift $((OPTIND -1))

if [ -z "$f_value" ] || [ -z "$q_value" ]; then
    echo "Both -f and -q parameters are required."
    exit 1
fi

python scripts/quantize.py -f "$f_value" -q "$q_value"