cp ../../../compile.sh ./
cp ../../../compile.py ./
bash compile.sh &> result.txt
python compile.py ./result.txt &> compiled_results.txt
