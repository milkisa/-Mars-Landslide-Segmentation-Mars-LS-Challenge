

docker run --gpus all --name milkisa --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm -t -u $(id -u):$(id -g) \
  -v "$(pwd)":"$(pwd)" \
  -v /media/datapart/milkisa:/mnt/data \
  -w "$(pwd)" \
  milkis/marsseg  \
  python /home/milkisayebasse/Mars/scripts/mars_train.py
