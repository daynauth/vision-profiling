export fine=false
export split=true

#options=(2 4 8 16 ultd)

#export MEM_SIZE=2

band=($(seq 8 23))
band+=(0)

run_once(){
    options=$1
    export MEM_SIZE=$options
    for i in "${band[@]}";do
        cd /home/roland/ML/vision-profiling/

        export BANDWIDTH=$((i*250))

        echo "Running tests with ${BANDWIDTH}mb/s of bandwidth"

            if [ $fine == true ] && [ $split == true ]; then
                echo "Running tests with fine-grained memory and split memory"
                cp optimizer/testcases/yolos-agx/1/prof_fine_split.csv optimizer/testcases/yolos-agx/1/prof.csv
                cp optimizer/testcases/yolos-agx/1/dep_fine_split.csv optimizer/testcases/yolos-agx/1/dep.csv
                dir=fine_split
            elif [ $fine == true ] && [ $split == false ]; then
                echo "Running tests with fine-grained memory and no split memory"
                cp optimizer/testcases/yolos-agx/1/prof_fine_no_split.csv optimizer/testcases/yolos-agx/1/prof.csv
                cp optimizer/testcases/yolos-agx/1/dep_fine_no_split.csv optimizer/testcases/yolos-agx/1/dep.csv
                dir=fine_no_split
            elif [ $fine == false ] && [ $split == true ]; then
                echo "Running tests with coarse-grained memory and split memory"
                cp optimizer/testcases/yolos-agx/1/prof_coarse_split.csv optimizer/testcases/yolos-agx/1/prof.csv
                cp optimizer/testcases/yolos-agx/1/dep_coarse_split.csv optimizer/testcases/yolos-agx/1/dep.csv
                dir=coarse_split
            elif [ $fine == false ] && [ $split == false ]; then
                echo "Running tests with coarse-grained memory and no split memory"
                cp optimizer/testcases/yolos-agx/1/prof_coarse_no_split.csv optimizer/testcases/yolos-agx/1/prof.csv
                cp optimizer/testcases/yolos-agx/1/dep_coarse_no_split.csv optimizer/testcases/yolos-agx/1/dep.csv
                dir=coarse_no_split
            else
                echo "Not a valid configuration"
                exit 1
            fi

            
            folder=results/${dir}/yolos-agx_${MEM_SIZE}gb

            mkdir -p $folder

            cd optimizer 
            if [ $MEM_SIZE == "ultd" ]; then
                echo "Running tests with unlimited memory"
                python opt_wrapper.py
            else
                echo "Running tests with ${MEM_SIZE}gb of memory"
                python opt_wrapper_mem.py
            fi


            #plot latency if i is 0
            if [ $i == 0 ]; then
                echo "Plotting latency"
                cd ..
                cd optimizer/PLT_latency
                python PYPLT_latency.py
                cp 1/yolos-agx.png ../../${folder}/latency.png
                cd ../..

            else
                cd ..
            fi


            cp optimizer/testcases/yolos-agx/1/part.csv ${folder}/part.csv

            #if i not 0, plot bandwidth
            if [ $i != 0 ]; then
                echo "Plotting bandwidth"
                cd visualization
                python render_encoder.py
                neato -Tpng encoder.dot -o graph.png
                mv graph.png ../${folder}/graph_${BANDWIDTH}.png
            else
                echo "Skipping bandwidth plot"
            fi

    done

}

option=(2 4 8 16 ultd)

for mem in "${option[@]}";do
    echo "Running tests with ${mem}gb of memory"
    run_once $mem
done