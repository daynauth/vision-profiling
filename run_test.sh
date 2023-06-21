export MEM_SIZE=16
folder=results/yolos-agx_${MEM_SIZE}gb
mkdir -p $folder

cd optimizer 
python opt_wrapper_mem.py 
cd ..
cd optimizer/PLT_latency
python PYPLT_latency.py
cp 1/yolos-agx.png ../../${folder}/latency.png
cd ../..

python /home/roland/ML/vision-profiling/NS-DOT-visualizers/colorer.py \
/home/roland/ML/vision-profiling/optimizer/testcases/yolos-agx/1/prof.csv \
/home/roland/ML/vision-profiling/optimizer/testcases/yolos-agx/1/dep.csv \
/home/roland/ML/vision-profiling/optimizer/testcases/yolos-agx/1/part.csv \
yolos

mv result_DOT_code_yolos.dot ${folder}/graph.dot

cp optimizer/testcases/yolos-agx/1/part.csv ${folder}/part.csv