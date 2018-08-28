.PHONY: prepare create_saved_models
.DEFAULT_GOAL := prepare

prepare: pull_images

pull_images:
	docker pull chie8842/tfserving
	docker pull chie8842/intern_mlops

create_data:
	# models及びsaved_modelsディレクトリ配下のデータを自分で作成するときのコマンド
	# Food-5Kデータのダウンロード
	curl -o /tmp/Food-5K.zip http://grebvm2.epfl.ch/lin/food/Food-5K.zip
	unzip /tmp/Food-5K.zip -d /tmp/Food-5K
	
	# データのコピー
	cp /tmp/Food-5K/training/1_* data/food-nonfood/train/food/
	cp /tmp/Food-5K/training/0_* data/food-nonfood/train/nonfood/
	
	# 不要なデータの削除
	rm -r /tmp/Food-5K*
	
	# flask実行用のモデル作成 （すでに作成済み）
	docker run -it --rm -v $(PWD):/work chie8842/intern_mlops \
    /usr/bin/python3 /work/src/retrain.py \
      --image_dir /work/data/food-nonfood/train/ \
      --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/feature_vector/2 \
      --output_graph=/work/models/retrained_mobilenet_v2_035_224.pb \
      --output_labels=/work/models/retrained_labels.txt
	
	# tensorflow-serving実行用モデル作成（すでに作成済み）
	docker run -it --rm -v $(PWD):/work chie8842/intern_mlops \
    /usr/bin/python3 /work/src/retrain.py \
      --image_dir /work/data/food-nonfood/train/ \
      --saved_model_dir=/work/saved_models/1/
	docker run -it --rm -v $(PWD):/work chie8842/intern_mlops \
    /usr/bin/python3 python /work/src/retrain.py \
      --image_dir /work/data/food-nonfood/train/ \
      --saved_model_dir=/work/saved_models/2/ \
      --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/feature_vector/1

docker-build:
# chie8842/intern_mlops, chie8842/tfserving Dockerイメージを自分で作成するときのコマンド
	docker build -t chie8842/intern_mlops -f docker/Dockerfile .
	docker build -t chie8842/tfserving -f docker/Dockerfile.tfs .
