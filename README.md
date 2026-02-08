docker build -t tf-gpu-train .


docker run --rm --gpus all `
  -v ${PWD}\Multi_alph:/app/Multi_alph `
  -v ${PWD}\outputs:/app/outputs `
  tf-gpu-train


  docker run --rm --gpus all -v ${PWD}\Multi_alph:/app/Multi_alph -v ${PWD}\outputs:/app/outputs tf-gpu-train