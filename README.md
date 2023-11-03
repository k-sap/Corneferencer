# Corneferencer http://zil.ipipan.waw.pl/Corneferencer

```
docker run --rm --gpus 0 -v ./Corneferencer/docker_corneferencer_volume/:/app/data corneferencer --input /app/data/one_text --output /app/data/one_text_pred  --model "/app/models/model_1190_features.h5"  -f "tei" --resolver all2all
```
