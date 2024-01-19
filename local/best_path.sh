dir=$2
symtab=$3
lattice-scale --acoustic-scale=`awk -v aa=$1 'BEGIN{print aa*0.1}'` "ark:gunzip -c $dir/lat.*.gz|" ark:- | \
	    lattice-best-path --word-symbol-table=$symtab ark:- ark,t:$dir/scoring/"$1".tra
