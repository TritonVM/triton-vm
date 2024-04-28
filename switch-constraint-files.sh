#!/usr/bin/bash

declare -a arr=("triton-vm/src/table/constraints.rs" "triton-vm/src/table/tasm_air_constraints.rs" "triton-vm/src/table/degree_lowering_table.rs")
for file in "${arr[@]}"
do
	cp $file $file.tmp
	cp $file.bkp $file
	cp $file.tmp $file.bkp
	rm $file.tmp
done

