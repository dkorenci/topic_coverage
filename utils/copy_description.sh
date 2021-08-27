# copy model description files (with topic labels) for specified number of topics
# to all the models produced by grid search with matching number of topics
# params: $1 number of topics $2 subfolder where model folder are

num=$1
subf=$2
for d in `ls $subf | grep T$num` 
do
	cp "description$num.xml" "$subf/$d/description.xml"
done
