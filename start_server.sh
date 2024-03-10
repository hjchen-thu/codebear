#!/bin/bash


s_value=""
l_value=""
t_value=""


while getopts ":s:l:t:" opt; do
  case ${opt} in
    s )
      s_value=$OPTARG
      ;;
    l )
      l_value=$OPTARG
      ;;
    t )
      t_value=$OPTARG
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


if [ -z "$s_value" ] || [ -z "$l_value" ] || [ -z "$t_value" ]; then
    echo "Parameters -s, -l, and -t are required."
    exit 1
fi


python scripts/serving.py -s "$s_value" -l "$l_value" -t "$t_value"
