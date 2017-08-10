CUDA班
====
正方行列の自乗を計算します。行列同士の掛け算は普通に計算するとO(N^3)かかります。これをCUDAコアを利用して並列で処理します。

サイズが小さい場合はCPUのみのほうが早いです。  
研究室のマシンのスペックだと1024\*1024だとCPUで0.5秒、CUDAで0.6秒くらい。  
2048\*2048だとCPUで15秒、CUDAで0.8秒くらい。

### 動作環境
**gcc**と**nvcc**が使えること。ちゃんとパスを通すこと。

### 使い方
makeすればCPUのみと、CUDA使用の両方をビルドできます。
```bash
$ make
```
ファイル名`cpu`がCPUのみを使用したもの、`cuda`がCUDAコアを使用したものです。

### オプション
#### サイズ指定
行列の大きさ指定できます。
```bash
$ make size=2048
```
この場合2048 * 2048の大きさの行列を計算します。

#### スレッド数指定
並列処理する際のブロックあたりのスレッド数も指定できます。  
最大で32までです。それ以上も指定できますが正しく計算されません。
```bash
$ make thread=32
```

#### 中身出力
-pオプションで行列の中身を表示させます
```bash
$ ./cpu -p
$ ./cuda -p
```

全部込みです。
```bash
$ make size=5 thread=1
```
上記のコマンドでビルドした`cuda`の実行結果は以下のようになります。
```
$ ./cuda -p
元の行列
 83  86  77  15  93 
 35  86  92  49  21 
 62  27  90  59  63 
 26  40  26  72  36 
 11  68  67  29  82 

2乗後の行列
 16086  23537  27854  13779  22542 
 13124  16278  21568  14304  14343 
 13898  16728  21113  13638  19293 
  7438  11706  12306  10112  10440 
  9103  15339  19381  11916  14440 

Use CUDA: THREAD=1, BLOCK=5
Matrix size: 5 * 5
matrix_ans[0][0]       = 16086
matrix_ans[SIZE][SIZE] = 14440

Processing time: 0.672068 sec
```

