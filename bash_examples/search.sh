# example 1: save number line of file example.log in total.log
total=$(cat example.log |wc -l > total.log)

# example 2: word count, print result
txt=$(tr -cs '[:alnum:]' '\n' < example.log | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -nr > wc.txt)