# 骨格線を利用したFunctional mapによる頂点間の対応付け

等長，非等長変換下における表面メッシュ間の対応付けを求める．
頂点間の対応付けを求めることで，テクスチャや変形の転写ができる．

転写例（左から右のモデルへの転写）
* テクスチャ（テクスチャ座標を転写）
  * <img src="https://github.com/metaout/curveSkeletonFmap/blob/master/example/dog.png" width="400"><img src="https://github.com/metaout/curveSkeletonFmap/blob/master/example/cat_to.png" width="400">

* アニメーション（変形行列とその重みを転写）
  * ![transfer4](/example/rhino2wolf.gif)
  * ![transfer5](/example/dog2dog.gif)
