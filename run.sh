protoc --python_out=. src/setup.proto
cp src/setup_pb2.py generator/
python src/parse_input.py --name=$1
python src/inter_chip.py --name=$1
python src/intra_chip.py --name=$1