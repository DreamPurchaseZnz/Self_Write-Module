# Self_Write-Module
## Process
Show the current state of training process

usage:
```
from process import Process
process = Process()
-
-
-
for epoch in range()
  -
  process.start_epoch()
  
  
  
  
  # print(38.7%|###-------| Epoch/Max_Epochs [elapsed: 39:19, left: 01:00:06, , name: value])
  process.format_meter(epoch, epochs, {"name": value})
```

## Learning-rate
To get different mode of learning rate decrease

usage:
```
from learning_rate import create_lr_schedule

lr_schedule = create_lr_schedule(lr_base=1E-3, decay_rate=0.1, decay_epochs=20000,
                                         truncated_epoch=30000, mode="exp")
                                         
lr_epoch = lr_schedule(epoch)
```

## utils
To save files or other things

usage:
```
from utils import pickle_save

```

## plot
usage:
```
from plot import Visualizer

vis = Visualizer()
    vis.mtsplot(loss, "Loss Tendency", name, dir)

```
